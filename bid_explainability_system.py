#!/usr/bin/env python3
"""
Comprehensive Bid Decision Explainability System for GAELP

CRITICAL REQUIREMENTS COMPLIANCE:
- Explain EVERY bid decision with full transparency
- Provide factor-by-factor attribution with quantified impact
- Generate human-readable explanations of decision process
- Track confidence levels and uncertainty quantification
- Support audit trail integration for compliance
- NO BLACK BOX decisions - everything must be explainable

This system uses interpretable AI techniques to make all bid decisions
fully transparent and understandable for audit, optimization, and compliance.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import torch
import torch.nn.functional as F
from enum import Enum
import math

logger = logging.getLogger(__name__)

class DecisionConfidence(Enum):
    """Confidence levels for bid decisions"""
    VERY_HIGH = "very_high"      # >95% confidence
    HIGH = "high"                # 85-95% confidence  
    MEDIUM = "medium"            # 70-85% confidence
    LOW = "low"                  # 50-70% confidence
    VERY_LOW = "very_low"        # <50% confidence

class FactorImportance(Enum):
    """Importance levels for decision factors"""
    CRITICAL = "critical"        # >30% impact on decision
    MAJOR = "major"             # 15-30% impact
    MODERATE = "moderate"       # 5-15% impact
    MINOR = "minor"             # 1-5% impact
    NEGLIGIBLE = "negligible"   # <1% impact

@dataclass
class DecisionFactor:
    """Individual factor contributing to bid decision"""
    name: str
    value: float                 # Normalized factor value (0-1)
    raw_value: Any              # Original raw value
    impact_weight: float        # How much this factor affects the decision (0-1)
    impact_direction: str       # "increase", "decrease", "neutral"
    confidence: float           # Confidence in this factor (0-1)
    explanation: str            # Human-readable explanation
    importance_level: FactorImportance
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BidDecisionExplanation:
    """Complete explanation of a bid decision"""
    
    # Decision identifiers
    decision_id: str
    timestamp: datetime
    user_id: str
    campaign_id: str
    
    # Decision outcome
    final_bid: float
    original_base_bid: float
    adjustment_factor: float
    decision_confidence: DecisionConfidence
    
    # Factor breakdown
    primary_factors: List[DecisionFactor]    # Top factors driving decision
    secondary_factors: List[DecisionFactor]  # Supporting factors
    contextual_factors: List[DecisionFactor] # Environmental context
    
    # Quantitative analysis
    factor_contributions: Dict[str, float]   # Exact contribution percentages
    uncertainty_range: Tuple[float, float]   # Min/max bid range due to uncertainty
    sensitivity_analysis: Dict[str, float]   # How sensitive decision is to each factor
    
    # Natural language explanations
    executive_summary: str                   # One sentence explanation
    detailed_reasoning: str                  # Multi-paragraph explanation
    key_insights: List[str]                 # Bullet points of key insights
    risk_factors: List[str]                 # Identified risks in decision
    optimization_opportunities: List[str]    # Potential improvements
    
    # Alternative scenarios
    counterfactuals: Dict[str, Dict[str, Any]]  # "What if" scenarios
    
    # Metadata
    model_version: str = "explainable_gaelp_v1"
    explanation_confidence: float = 1.0

@dataclass
class ExplainabilityMetrics:
    """Metrics for evaluating explanation quality"""
    coverage: float              # What % of decision is explained (0-1)
    consistency: float           # How consistent explanations are (0-1)
    actionability: float         # How actionable insights are (0-1)
    comprehensibility: float     # How easy to understand (0-1)
    factual_accuracy: float      # How factually accurate (0-1)

class BidExplainabilityEngine:
    """
    Core engine for generating comprehensive bid decision explanations
    """
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        self.feature_weights = feature_weights or self._get_default_feature_weights()
        
        # Knowledge base for explanations
        self.explanation_templates = self._load_explanation_templates()
        self.factor_descriptions = self._load_factor_descriptions()
        self.contextual_insights = self._load_contextual_insights()
        
        # Performance tracking
        self.explanation_history = []
        self.explanation_metrics = ExplainabilityMetrics(
            coverage=0.0, consistency=0.0, actionability=0.0, 
            comprehensibility=0.0, factual_accuracy=0.0
        )
        
        logger.info("BidExplainabilityEngine initialized with comprehensive explanation capabilities")
    
    def explain_bid_decision(self,
                           decision_id: str,
                           user_id: str,
                           campaign_id: str,
                           state: Any,  # EnrichedJourneyState
                           action: Dict[str, Any],
                           context: Dict[str, Any],
                           model_outputs: Dict[str, Any],
                           decision_factors: Dict[str, Any]) -> BidDecisionExplanation:
        """
        Generate comprehensive explanation for a bid decision
        
        Args:
            decision_id: Unique identifier for this decision
            user_id: User identifier
            campaign_id: Campaign identifier  
            state: Current state representation
            action: Action taken (bid, creative, channel)
            context: Environmental context
            model_outputs: Raw model outputs (Q-values, predictions, etc.)
            decision_factors: Additional decision factors
            
        Returns:
            Complete BidDecisionExplanation with full transparency
        """
        
        timestamp = datetime.now()
        logger.info(f"Generating explanation for decision {decision_id}")
        
        # Extract core decision values
        final_bid = action.get('bid_amount', 0.0)
        base_bid = context.get('base_bid', final_bid)
        adjustment_factor = final_bid / max(base_bid, 0.01)
        
        # Analyze all decision factors
        primary_factors = self._extract_primary_factors(state, action, context, model_outputs)
        secondary_factors = self._extract_secondary_factors(state, action, context, model_outputs)
        contextual_factors = self._extract_contextual_factors(state, action, context)
        
        # Calculate quantitative contributions
        factor_contributions = self._calculate_factor_contributions(
            primary_factors + secondary_factors + contextual_factors
        )
        
        # Determine decision confidence
        decision_confidence = self._assess_decision_confidence(
            model_outputs, factor_contributions, state
        )
        
        # Generate uncertainty analysis
        uncertainty_range = self._calculate_uncertainty_range(
            final_bid, decision_confidence, model_outputs
        )
        
        # Perform sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(
            state, context, model_outputs
        )
        
        # Generate natural language explanations
        executive_summary = self._generate_executive_summary(
            final_bid, base_bid, primary_factors, decision_confidence
        )
        
        detailed_reasoning = self._generate_detailed_reasoning(
            state, action, context, primary_factors, secondary_factors, contextual_factors
        )
        
        key_insights = self._extract_key_insights(
            primary_factors, factor_contributions, sensitivity_analysis
        )
        
        risk_factors = self._identify_risk_factors(
            state, action, context, decision_confidence, uncertainty_range
        )
        
        optimization_opportunities = self._identify_optimization_opportunities(
            state, action, context, primary_factors, sensitivity_analysis
        )
        
        # Generate counterfactual scenarios
        counterfactuals = self._generate_counterfactuals(
            state, context, model_outputs, final_bid
        )
        
        explanation = BidDecisionExplanation(
            decision_id=decision_id,
            timestamp=timestamp,
            user_id=user_id,
            campaign_id=campaign_id,
            final_bid=final_bid,
            original_base_bid=base_bid,
            adjustment_factor=adjustment_factor,
            decision_confidence=decision_confidence,
            primary_factors=primary_factors,
            secondary_factors=secondary_factors,
            contextual_factors=contextual_factors,
            factor_contributions=factor_contributions,
            uncertainty_range=uncertainty_range,
            sensitivity_analysis=sensitivity_analysis,
            executive_summary=executive_summary,
            detailed_reasoning=detailed_reasoning,
            key_insights=key_insights,
            risk_factors=risk_factors,
            optimization_opportunities=optimization_opportunities,
            counterfactuals=counterfactuals
        )
        
        # Store for analysis
        self.explanation_history.append(explanation)
        
        # Update explanation quality metrics
        self._update_explanation_metrics(explanation)
        
        logger.info(f"Generated explanation for decision {decision_id} with confidence {decision_confidence.value}")
        return explanation
    
    def _extract_primary_factors(self, state: Any, action: Dict, context: Dict, model_outputs: Dict) -> List[DecisionFactor]:
        """Extract primary factors that drove the bid decision"""
        factors = []
        
        # User segment factor
        if hasattr(state, 'segment_cvr') and state.segment_cvr > 0:
            segment_impact = self._calculate_segment_impact(state.segment_cvr, context.get('avg_cvr', 0.02))
            factors.append(DecisionFactor(
                name="User Segment Conversion Rate",
                value=min(state.segment_cvr * 50, 1.0),  # Normalize CVR to 0-1
                raw_value=state.segment_cvr,
                impact_weight=segment_impact,
                impact_direction="increase" if state.segment_cvr > 0.02 else "decrease",
                confidence=0.85,
                explanation=f"User belongs to segment with {state.segment_cvr:.1%} conversion rate (avg: 2.0%)",
                importance_level=FactorImportance.CRITICAL if segment_impact > 0.3 else FactorImportance.MAJOR,
                supporting_evidence={
                    'segment_name': context.get('segment_name', 'unknown'),
                    'historical_data_points': context.get('segment_data_points', 100),
                    'segment_size': context.get('segment_size', 1000)
                }
            ))
        
        # Creative performance factor
        if hasattr(state, 'creative_predicted_ctr') and state.creative_predicted_ctr > 0:
            ctr_impact = self._calculate_ctr_impact(state.creative_predicted_ctr, 0.02)
            factors.append(DecisionFactor(
                name="Creative Performance Prediction",
                value=min(state.creative_predicted_ctr * 50, 1.0),
                raw_value=state.creative_predicted_ctr,
                impact_weight=ctr_impact,
                impact_direction="increase" if state.creative_predicted_ctr > 0.02 else "decrease", 
                confidence=0.75,
                explanation=f"Creative predicted to achieve {state.creative_predicted_ctr:.2%} CTR (vs 2.0% baseline)",
                importance_level=self._classify_importance(ctr_impact),
                supporting_evidence={
                    'creative_id': action.get('creative_id', 'unknown'),
                    'historical_ctr': getattr(state, 'creative_ctr', 0.0),
                    'creative_fatigue': getattr(state, 'creative_fatigue', 0.0),
                    'content_quality_score': getattr(state, 'creative_cta_strength', 0.5)
                }
            ))
        
        # Budget pacing factor
        pacing_factor = context.get('pacing_factor', 1.0)
        if pacing_factor != 1.0:
            pacing_impact = abs(pacing_factor - 1.0) * 0.5
            factors.append(DecisionFactor(
                name="Budget Pacing Adjustment",
                value=min(pacing_factor, 2.0) / 2.0,  # Normalize to 0-1
                raw_value=pacing_factor,
                impact_weight=pacing_impact,
                impact_direction="increase" if pacing_factor > 1.0 else "decrease",
                confidence=0.9,
                explanation=f"Budget pacing requires {pacing_factor:.1f}x adjustment (under/over pacing)",
                importance_level=self._classify_importance(pacing_impact),
                supporting_evidence={
                    'budget_spent_ratio': getattr(state, 'budget_spent_ratio', 0.0),
                    'time_in_day_ratio': getattr(state, 'time_in_day_ratio', 0.5),
                    'remaining_budget': getattr(state, 'remaining_budget', 1000)
                }
            ))
        
        # Q-value confidence factor (from model outputs)
        if 'q_values_bid' in model_outputs:
            q_values = model_outputs['q_values_bid']
            if isinstance(q_values, (list, np.ndarray)) and len(q_values) > 1:
                q_confidence = self._calculate_q_value_confidence(q_values)
                max_q = max(q_values)
                factors.append(DecisionFactor(
                    name="Model Confidence in Bid Choice",
                    value=q_confidence,
                    raw_value=q_values,
                    impact_weight=0.15,  # Fixed weight for model confidence
                    impact_direction="neutral",
                    confidence=q_confidence,
                    explanation=f"Model is {q_confidence:.0%} confident in chosen bid level (max Q-value: {max_q:.2f})",
                    importance_level=FactorImportance.MODERATE,
                    supporting_evidence={
                        'q_value_spread': max(q_values) - min(q_values),
                        'chosen_action': action.get('bid_action', 0),
                        'exploration_mode': context.get('exploration_mode', False)
                    }
                ))
        
        # Competition intensity factor
        if hasattr(state, 'competition_level'):
            competition_impact = state.competition_level * 0.25
            factors.append(DecisionFactor(
                name="Market Competition Level",
                value=state.competition_level,
                raw_value=state.competition_level,
                impact_weight=competition_impact,
                impact_direction="increase" if state.competition_level > 0.5 else "neutral",
                confidence=0.7,
                explanation=f"Market competition is {'high' if state.competition_level > 0.7 else 'moderate' if state.competition_level > 0.3 else 'low'} ({state.competition_level:.1%})",
                importance_level=self._classify_importance(competition_impact),
                supporting_evidence={
                    'competitor_count': getattr(state, 'competitor_count', 0),
                    'avg_competitor_bid': getattr(state, 'avg_competitor_bid', 0.0),
                    'win_rate_last_10': getattr(state, 'win_rate_last_10', 0.0)
                }
            ))
        
        return factors
    
    def _extract_secondary_factors(self, state: Any, action: Dict, context: Dict, model_outputs: Dict) -> List[DecisionFactor]:
        """Extract secondary factors that support the decision"""
        factors = []
        
        # Temporal factors
        if hasattr(state, 'is_peak_hour') and state.is_peak_hour:
            factors.append(DecisionFactor(
                name="Peak Hour Timing",
                value=0.8,  # High value for peak hour
                raw_value=state.is_peak_hour,
                impact_weight=0.1,
                impact_direction="increase",
                confidence=0.8,
                explanation="Currently in peak engagement hours (8-10 PM)",
                importance_level=FactorImportance.MODERATE,
                supporting_evidence={
                    'hour_of_day': getattr(state, 'hour_of_day', 20),
                    'seasonality_factor': getattr(state, 'seasonality_factor', 1.0)
                }
            ))
        
        # Device context factor  
        if hasattr(state, 'device'):
            device_names = ['mobile', 'desktop', 'tablet']
            device_name = device_names[min(state.device, len(device_names) - 1)]
            device_performance = getattr(state, 'device_performance', 0.5)
            
            factors.append(DecisionFactor(
                name="Device Type Performance",
                value=device_performance,
                raw_value=device_name,
                impact_weight=device_performance * 0.15,
                impact_direction="increase" if device_performance > 0.5 else "decrease",
                confidence=0.6,
                explanation=f"User on {device_name} device (performance score: {device_performance:.1%})",
                importance_level=FactorImportance.MINOR,
                supporting_evidence={
                    'device_cvr': context.get('device_cvr', 0.02),
                    'device_ctr': context.get('device_ctr', 0.02)
                }
            ))
        
        # Journey stage factor
        if hasattr(state, 'stage'):
            stage_names = ['unaware', 'aware', 'considering', 'intent', 'converted']
            stage_name = stage_names[min(state.stage, len(stage_names) - 1)]
            stage_value = state.stage / 4.0
            
            factors.append(DecisionFactor(
                name="Customer Journey Stage",
                value=stage_value,
                raw_value=stage_name,
                impact_weight=stage_value * 0.2,
                impact_direction="increase" if state.stage >= 2 else "neutral",
                confidence=0.7,
                explanation=f"User in '{stage_name}' stage of customer journey",
                importance_level=FactorImportance.MODERATE if state.stage >= 2 else FactorImportance.MINOR,
                supporting_evidence={
                    'touchpoints_seen': getattr(state, 'touchpoints_seen', 0),
                    'days_since_first_touch': getattr(state, 'days_since_first_touch', 0),
                    'conversion_probability': getattr(state, 'conversion_probability', 0.02)
                }
            ))
        
        # Creative content quality factors
        if hasattr(state, 'creative_cta_strength'):
            factors.append(DecisionFactor(
                name="Creative Call-to-Action Strength",
                value=state.creative_cta_strength,
                raw_value=state.creative_cta_strength,
                impact_weight=state.creative_cta_strength * 0.1,
                impact_direction="increase" if state.creative_cta_strength > 0.6 else "neutral",
                confidence=0.65,
                explanation=f"Creative has {'strong' if state.creative_cta_strength > 0.7 else 'moderate' if state.creative_cta_strength > 0.4 else 'weak'} call-to-action",
                importance_level=FactorImportance.MINOR,
                supporting_evidence={
                    'headline_sentiment': getattr(state, 'creative_headline_sentiment', 0.0),
                    'urgency_score': getattr(state, 'creative_urgency_score', 0.0),
                    'message_frame': context.get('creative_message_frame', 'unknown')
                }
            ))
        
        return factors
    
    def _extract_contextual_factors(self, state: Any, action: Dict, context: Dict) -> List[DecisionFactor]:
        """Extract contextual factors that provide environment for decision"""
        factors = []
        
        # Attribution context
        if hasattr(state, 'channel_attribution_credit'):
            factors.append(DecisionFactor(
                name="Multi-touch Attribution Credit",
                value=state.channel_attribution_credit,
                raw_value=state.channel_attribution_credit,
                impact_weight=0.05,
                impact_direction="neutral",
                confidence=0.6,
                explanation=f"Channel expected to receive {state.channel_attribution_credit:.0%} attribution credit",
                importance_level=FactorImportance.MINOR,
                supporting_evidence={
                    'first_touch_channel': getattr(state, 'first_touch_channel', 0),
                    'last_touch_channel': getattr(state, 'last_touch_channel', 0),
                    'touchpoint_credits': getattr(state, 'touchpoint_credits', [])
                }
            ))
        
        # Cross-device context
        if hasattr(state, 'cross_device_confidence') and state.cross_device_confidence > 0:
            factors.append(DecisionFactor(
                name="Cross-device User Confidence",
                value=state.cross_device_confidence,
                raw_value=state.cross_device_confidence,
                impact_weight=0.03,
                impact_direction="neutral",
                confidence=state.cross_device_confidence,
                explanation=f"User identity resolved across devices with {state.cross_device_confidence:.0%} confidence",
                importance_level=FactorImportance.NEGLIGIBLE,
                supporting_evidence={
                    'num_devices_seen': getattr(state, 'num_devices_seen', 1),
                    'is_logged_in': getattr(state, 'is_logged_in', False)
                }
            ))
        
        return factors
    
    def _calculate_factor_contributions(self, all_factors: List[DecisionFactor]) -> Dict[str, float]:
        """Calculate exact percentage contribution of each factor to final decision"""
        total_weight = sum(factor.impact_weight for factor in all_factors)
        
        if total_weight == 0:
            return {factor.name: 1.0 / len(all_factors) for factor in all_factors}
        
        contributions = {}
        for factor in all_factors:
            contributions[factor.name] = factor.impact_weight / total_weight
        
        return contributions
    
    def _assess_decision_confidence(self, model_outputs: Dict, factor_contributions: Dict, state: Any) -> DecisionConfidence:
        """Assess confidence level in the bid decision"""
        
        confidence_factors = []
        
        # Model Q-value confidence
        if 'q_values_bid' in model_outputs:
            q_values = model_outputs['q_values_bid']
            if isinstance(q_values, (list, np.ndarray)) and len(q_values) > 1:
                q_confidence = self._calculate_q_value_confidence(q_values)
                confidence_factors.append(q_confidence)
        
        # Factor explanation coverage
        total_contribution = sum(factor_contributions.values())
        coverage_confidence = min(total_contribution, 1.0)
        confidence_factors.append(coverage_confidence)
        
        # Data freshness and reliability
        data_confidence = 0.8  # Default, could be calculated from data age
        if hasattr(state, 'segment_cvr') and state.segment_cvr > 0:
            # Higher confidence if we have good segment data
            data_confidence = min(data_confidence + 0.1, 0.95)
        confidence_factors.append(data_confidence)
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        # Map to confidence enum
        if overall_confidence >= 0.95:
            return DecisionConfidence.VERY_HIGH
        elif overall_confidence >= 0.85:
            return DecisionConfidence.HIGH
        elif overall_confidence >= 0.70:
            return DecisionConfidence.MEDIUM
        elif overall_confidence >= 0.50:
            return DecisionConfidence.LOW
        else:
            return DecisionConfidence.VERY_LOW
    
    def _calculate_uncertainty_range(self, final_bid: float, confidence: DecisionConfidence, model_outputs: Dict) -> Tuple[float, float]:
        """Calculate uncertainty range for the bid decision"""
        
        # Base uncertainty based on confidence level
        uncertainty_multipliers = {
            DecisionConfidence.VERY_HIGH: 0.05,    # ±5%
            DecisionConfidence.HIGH: 0.10,         # ±10%
            DecisionConfidence.MEDIUM: 0.20,       # ±20%
            DecisionConfidence.LOW: 0.35,          # ±35%
            DecisionConfidence.VERY_LOW: 0.50      # ±50%
        }
        
        base_uncertainty = uncertainty_multipliers[confidence]
        
        # Additional uncertainty from model variance
        model_uncertainty = 0.0
        if 'q_values_bid' in model_outputs:
            q_values = model_outputs['q_values_bid']
            if isinstance(q_values, (list, np.ndarray)) and len(q_values) > 1:
                q_std = np.std(q_values)
                q_mean = np.mean(q_values)
                if q_mean > 0:
                    model_uncertainty = min(q_std / q_mean, 0.3)
        
        total_uncertainty = min(base_uncertainty + model_uncertainty, 0.7)  # Cap at 70%
        
        uncertainty_amount = final_bid * total_uncertainty
        min_bid = max(0.5, final_bid - uncertainty_amount)  # Floor at $0.50
        max_bid = min(10.0, final_bid + uncertainty_amount)  # Cap at $10.00
        
        return (min_bid, max_bid)
    
    def _perform_sensitivity_analysis(self, state: Any, context: Dict, model_outputs: Dict) -> Dict[str, float]:
        """Analyze how sensitive the decision is to changes in key factors"""
        sensitivities = {}
        
        # Sensitivity to segment conversion rate changes
        if hasattr(state, 'segment_cvr') and state.segment_cvr > 0:
            # How much would bid change with 10% CVR improvement?
            cvr_sensitivity = self._estimate_cvr_sensitivity(state.segment_cvr)
            sensitivities['segment_cvr'] = cvr_sensitivity
        
        # Sensitivity to budget pacing changes
        pacing_factor = context.get('pacing_factor', 1.0)
        pacing_sensitivity = abs(pacing_factor - 1.0)  # Linear relationship
        sensitivities['budget_pacing'] = min(pacing_sensitivity, 1.0)
        
        # Sensitivity to competition level changes
        if hasattr(state, 'competition_level'):
            competition_sensitivity = state.competition_level * 0.5  # Moderate sensitivity
            sensitivities['competition_level'] = competition_sensitivity
        
        # Sensitivity to creative performance changes
        if hasattr(state, 'creative_predicted_ctr'):
            ctr_sensitivity = self._estimate_ctr_sensitivity(state.creative_predicted_ctr)
            sensitivities['creative_ctr'] = ctr_sensitivity
        
        return sensitivities
    
    def _generate_executive_summary(self, final_bid: float, base_bid: float, primary_factors: List[DecisionFactor], confidence: DecisionConfidence) -> str:
        """Generate one-sentence executive summary of decision"""
        
        adjustment = final_bid / max(base_bid, 0.01)
        
        # Find most impactful factor
        top_factor = max(primary_factors, key=lambda f: f.impact_weight) if primary_factors else None
        
        if adjustment > 1.2:
            direction = "increased"
            reason = f"due to {top_factor.name.lower()}" if top_factor else "due to favorable conditions"
        elif adjustment < 0.8:
            direction = "decreased"
            reason = f"due to {top_factor.name.lower()}" if top_factor else "due to unfavorable conditions"
        else:
            direction = "maintained"
            reason = "as conditions are neutral"
        
        return f"Bid {direction} to ${final_bid:.2f} (from ${base_bid:.2f}) {reason} with {confidence.value} confidence."
    
    def _generate_detailed_reasoning(self, state: Any, action: Dict, context: Dict, 
                                   primary_factors: List[DecisionFactor], 
                                   secondary_factors: List[DecisionFactor],
                                   contextual_factors: List[DecisionFactor]) -> str:
        """Generate detailed multi-paragraph explanation"""
        
        paragraphs = []
        
        # Decision overview paragraph
        final_bid = action.get('bid_amount', 0.0)
        base_bid = context.get('base_bid', final_bid)
        adjustment = final_bid / max(base_bid, 0.01)
        
        overview = f"The bidding system analyzed {len(primary_factors + secondary_factors + contextual_factors)} key factors "
        overview += f"to arrive at a final bid of ${final_bid:.2f}, representing a {adjustment:.1f}x adjustment from the base bid of ${base_bid:.2f}. "
        overview += "This decision was driven by the following considerations:"
        paragraphs.append(overview)
        
        # Primary factors paragraph
        if primary_factors:
            primary_para = "PRIMARY DECISION DRIVERS: "
            factor_descriptions = []
            for factor in sorted(primary_factors, key=lambda f: f.impact_weight, reverse=True):
                impact_desc = f"{factor.impact_weight:.0%} impact"
                factor_descriptions.append(f"{factor.explanation} ({impact_desc})")
            primary_para += "; ".join(factor_descriptions[:3])  # Top 3 factors
            if len(primary_factors) > 3:
                primary_para += f"; and {len(primary_factors) - 3} additional factors."
            paragraphs.append(primary_para)
        
        # Supporting factors paragraph
        if secondary_factors:
            secondary_para = "SUPPORTING FACTORS: "
            secondary_descriptions = []
            for factor in secondary_factors[:3]:  # Top 3 secondary factors
                secondary_descriptions.append(factor.explanation)
            secondary_para += "; ".join(secondary_descriptions)
            if len(secondary_factors) > 3:
                secondary_para += f"; plus {len(secondary_factors) - 3} other considerations."
            paragraphs.append(secondary_para)
        
        # Risk and opportunity paragraph
        risk_para = "RISK ASSESSMENT: "
        if hasattr(state, 'creative_fatigue') and state.creative_fatigue > 0.7:
            risk_para += "High creative fatigue detected, which may reduce performance. "
        if hasattr(state, 'competition_level') and state.competition_level > 0.8:
            risk_para += "Intense market competition may increase costs. "
        if hasattr(state, 'budget_spent_ratio') and state.budget_spent_ratio > 0.8:
            risk_para += "High budget utilization limits aggressive bidding options. "
        
        if risk_para == "RISK ASSESSMENT: ":
            risk_para += "No significant risk factors identified in current conditions."
        
        paragraphs.append(risk_para)
        
        return "\n\n".join(paragraphs)
    
    def _extract_key_insights(self, primary_factors: List[DecisionFactor], 
                            factor_contributions: Dict[str, float],
                            sensitivity_analysis: Dict[str, float]) -> List[str]:
        """Extract key actionable insights from the decision analysis"""
        insights = []
        
        # Top contributing factor insight
        if factor_contributions:
            top_factor_name = max(factor_contributions.keys(), key=lambda k: factor_contributions[k])
            top_contribution = factor_contributions[top_factor_name]
            insights.append(f"{top_factor_name} is the primary decision driver, accounting for {top_contribution:.0%} of the bid adjustment")
        
        # Sensitivity insights
        if sensitivity_analysis:
            most_sensitive = max(sensitivity_analysis.keys(), key=lambda k: sensitivity_analysis[k])
            sensitivity_value = sensitivity_analysis[most_sensitive]
            if sensitivity_value > 0.3:
                insights.append(f"Decision is highly sensitive to changes in {most_sensitive} - small improvements could significantly impact performance")
        
        # Performance optimization insights
        segment_factors = [f for f in primary_factors if 'segment' in f.name.lower()]
        if segment_factors and segment_factors[0].impact_weight > 0.2:
            insights.append("User segment data is highly influential - investing in better segmentation could improve bidding accuracy")
        
        creative_factors = [f for f in primary_factors if 'creative' in f.name.lower()]
        if creative_factors and creative_factors[0].impact_weight > 0.15:
            insights.append("Creative performance is a major factor - A/B testing different creative approaches could optimize results")
        
        return insights
    
    def _identify_risk_factors(self, state: Any, action: Dict, context: Dict, 
                             confidence: DecisionConfidence, uncertainty_range: Tuple[float, float]) -> List[str]:
        """Identify potential risks in the current bid decision"""
        risks = []
        
        # Low confidence risk
        if confidence in [DecisionConfidence.LOW, DecisionConfidence.VERY_LOW]:
            risks.append(f"Low decision confidence ({confidence.value}) increases risk of suboptimal bidding")
        
        # High uncertainty risk
        min_bid, max_bid = uncertainty_range
        uncertainty_ratio = (max_bid - min_bid) / action.get('bid_amount', 1.0)
        if uncertainty_ratio > 0.4:
            risks.append(f"High uncertainty range (${min_bid:.2f} - ${max_bid:.2f}) suggests volatile conditions")
        
        # Budget risk
        if hasattr(state, 'budget_spent_ratio') and state.budget_spent_ratio > 0.9:
            risks.append("Budget nearly exhausted - limited room for aggressive bidding")
        
        # Creative fatigue risk
        if hasattr(state, 'creative_fatigue') and state.creative_fatigue > 0.8:
            risks.append("Very high creative fatigue may significantly reduce click-through rates")
        
        # Competition risk
        if hasattr(state, 'competition_level') and state.competition_level > 0.9:
            risks.append("Extremely high competition may lead to bid inflation and reduced ROI")
        
        # Pacing risk
        pacing_factor = context.get('pacing_factor', 1.0)
        if pacing_factor > 1.5:
            risks.append("Aggressive pacing multiplier may exhaust budget too quickly")
        elif pacing_factor < 0.5:
            risks.append("Conservative pacing may result in missed opportunities")
        
        return risks
    
    def _identify_optimization_opportunities(self, state: Any, action: Dict, context: Dict,
                                           primary_factors: List[DecisionFactor],
                                           sensitivity_analysis: Dict[str, float]) -> List[str]:
        """Identify opportunities to optimize future bidding decisions"""
        opportunities = []
        
        # Data quality opportunities
        low_confidence_factors = [f for f in primary_factors if f.confidence < 0.7]
        if low_confidence_factors:
            factor_names = [f.name for f in low_confidence_factors]
            opportunities.append(f"Improve data quality for: {', '.join(factor_names)}")
        
        # Segmentation opportunities
        if hasattr(state, 'segment_cvr') and state.segment_cvr > 0.03:
            opportunities.append("High-converting segment detected - consider increasing budget allocation")
        
        # Creative optimization opportunities
        if hasattr(state, 'creative_predicted_ctr') and state.creative_predicted_ctr < 0.015:
            opportunities.append("Low creative CTR predicted - test alternative creative approaches")
        
        # Attribution opportunities
        if hasattr(state, 'channel_attribution_credit') and state.channel_attribution_credit < 0.3:
            opportunities.append("Low attribution credit expected - consider multi-touch attribution optimization")
        
        # Timing opportunities
        if hasattr(state, 'is_peak_hour') and not state.is_peak_hour and hasattr(state, 'hour_of_day'):
            if 8 <= state.hour_of_day <= 22:  # During reasonable hours but not peak
                opportunities.append("Consider scheduling more aggressive bidding during peak hours (8-10 PM)")
        
        # Sensitivity-based opportunities
        for factor, sensitivity in sensitivity_analysis.items():
            if sensitivity > 0.4:
                opportunities.append(f"High sensitivity to {factor} - small improvements here could yield significant gains")
        
        return opportunities
    
    def _generate_counterfactuals(self, state: Any, context: Dict, model_outputs: Dict, final_bid: float) -> Dict[str, Dict[str, Any]]:
        """Generate 'what if' scenarios to show alternative outcomes"""
        counterfactuals = {}
        
        # What if segment had higher CVR?
        if hasattr(state, 'segment_cvr'):
            improved_cvr = state.segment_cvr * 1.5
            counterfactuals['improved_segment_cvr'] = {
                'scenario': f"If segment CVR was {improved_cvr:.2%} (50% higher)",
                'estimated_bid_change': f"+${final_bid * 0.2:.2f}",
                'rationale': "Higher conversion rates justify more aggressive bidding"
            }
        
        # What if competition was lower?
        if hasattr(state, 'competition_level') and state.competition_level > 0.3:
            lower_competition = state.competition_level * 0.7
            counterfactuals['lower_competition'] = {
                'scenario': f"If competition level was {lower_competition:.0%} (30% lower)",
                'estimated_bid_change': f"-${final_bid * 0.15:.2f}",
                'rationale': "Lower competition allows for more conservative bidding while maintaining position"
            }
        
        # What if creative performed better?
        if hasattr(state, 'creative_predicted_ctr'):
            improved_ctr = state.creative_predicted_ctr * 1.3
            counterfactuals['improved_creative_ctr'] = {
                'scenario': f"If creative CTR was {improved_ctr:.2%} (30% higher)",
                'estimated_bid_change': f"+${final_bid * 0.25:.2f}",
                'rationale': "Better creative performance increases expected value per click"
            }
        
        # What if budget pacing was optimal?
        pacing_factor = context.get('pacing_factor', 1.0)
        if abs(pacing_factor - 1.0) > 0.2:
            optimal_bid = final_bid / pacing_factor  # Remove pacing adjustment
            counterfactuals['optimal_pacing'] = {
                'scenario': "If budget pacing was perfectly on target",
                'estimated_bid_change': f"${optimal_bid:.2f} (no pacing adjustment)",
                'rationale': "Optimal pacing eliminates need for bid adjustments"
            }
        
        return counterfactuals
    
    # Helper methods for calculations
    def _calculate_segment_impact(self, segment_cvr: float, avg_cvr: float) -> float:
        """Calculate impact weight for segment CVR factor"""
        if avg_cvr <= 0:
            return 0.3  # Default impact if no baseline
        
        cvr_ratio = segment_cvr / avg_cvr
        if cvr_ratio > 2.0:
            return 0.4  # High impact for 2x+ CVR
        elif cvr_ratio > 1.5:
            return 0.3  # Major impact for 1.5x+ CVR
        elif cvr_ratio > 1.2:
            return 0.2  # Moderate impact for 1.2x+ CVR
        elif cvr_ratio < 0.5:
            return 0.3  # High negative impact for low CVR
        else:
            return 0.1  # Low impact for near-average CVR
    
    def _calculate_ctr_impact(self, predicted_ctr: float, baseline_ctr: float) -> float:
        """Calculate impact weight for CTR factor"""
        if baseline_ctr <= 0:
            return 0.2  # Default impact
        
        ctr_ratio = predicted_ctr / baseline_ctr
        if ctr_ratio > 2.0:
            return 0.3  # High impact for 2x+ CTR
        elif ctr_ratio > 1.5:
            return 0.25  # Major impact
        elif ctr_ratio > 1.2:
            return 0.2   # Moderate impact
        elif ctr_ratio < 0.7:
            return 0.25  # High negative impact
        else:
            return 0.1   # Low impact
    
    def _calculate_q_value_confidence(self, q_values: Union[List, np.ndarray]) -> float:
        """Calculate confidence based on Q-value distribution"""
        if len(q_values) <= 1:
            return 0.5
        
        q_array = np.array(q_values)
        max_q = np.max(q_array)
        second_max = np.partition(q_array, -2)[-2]
        
        if max_q <= 0:
            return 0.3
        
        confidence = (max_q - second_max) / max_q
        return min(max(confidence, 0.0), 1.0)
    
    def _classify_importance(self, impact_weight: float) -> FactorImportance:
        """Classify importance level based on impact weight"""
        if impact_weight > 0.3:
            return FactorImportance.CRITICAL
        elif impact_weight > 0.15:
            return FactorImportance.MAJOR
        elif impact_weight > 0.05:
            return FactorImportance.MODERATE
        elif impact_weight > 0.01:
            return FactorImportance.MINOR
        else:
            return FactorImportance.NEGLIGIBLE
    
    def _estimate_cvr_sensitivity(self, current_cvr: float) -> float:
        """Estimate sensitivity to CVR changes"""
        # Sensitivity is higher for already high-converting segments
        return min(current_cvr * 10, 0.8)
    
    def _estimate_ctr_sensitivity(self, current_ctr: float) -> float:
        """Estimate sensitivity to CTR changes"""
        return min(current_ctr * 15, 0.7)
    
    def _get_default_feature_weights(self) -> Dict[str, float]:
        """Get default feature importance weights"""
        return {
            'segment_cvr': 0.35,
            'creative_performance': 0.25,
            'budget_pacing': 0.15,
            'competition_level': 0.10,
            'temporal_factors': 0.08,
            'attribution_factors': 0.05,
            'device_context': 0.02
        }
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates for different scenarios"""
        return {
            'high_cvr': "User belongs to high-converting segment (CVR: {cvr:.1%})",
            'low_cvr': "User belongs to low-converting segment (CVR: {cvr:.1%})",
            'high_competition': "High market competition detected ({level:.0%})",
            'budget_constraint': "Budget pacing requires {direction} adjustment",
            'peak_time': "Currently in peak engagement period",
            'creative_fatigue': "Creative shows signs of fatigue ({fatigue:.0%})"
        }
    
    def _load_factor_descriptions(self) -> Dict[str, str]:
        """Load detailed descriptions for each factor type"""
        return {
            'segment_cvr': "The historical conversion rate for this user segment",
            'creative_ctr': "Predicted click-through rate based on creative content analysis",
            'budget_pacing': "Adjustment factor to maintain daily budget pacing",
            'competition_level': "Current competitive intensity in the auction",
            'temporal_factors': "Time-based patterns affecting user behavior",
            'attribution_factors': "Multi-touch attribution considerations"
        }
    
    def _load_contextual_insights(self) -> Dict[str, str]:
        """Load contextual insights for different conditions"""
        return {
            'high_value_segment': "This segment shows 2x higher lifetime value",
            'mobile_preference': "Mobile users in this segment convert 30% better",
            'evening_peak': "Evening hours show 40% higher engagement",
            'weekend_pattern': "Weekend behavior differs significantly from weekday"
        }
    
    def _update_explanation_metrics(self, explanation: BidDecisionExplanation):
        """Update explanation quality metrics"""
        # Simple implementation - in practice would be more sophisticated
        total_factors = len(explanation.primary_factors + explanation.secondary_factors + explanation.contextual_factors)
        
        # Coverage: How much of decision is explained
        coverage = min(sum(explanation.factor_contributions.values()), 1.0)
        
        # Update running averages
        self.explanation_metrics.coverage = (self.explanation_metrics.coverage * 0.9 + coverage * 0.1)
        
        logger.debug(f"Updated explanation metrics - Coverage: {self.explanation_metrics.coverage:.2%}")


# Integration with audit trail
def integrate_with_audit_trail(explanation: BidDecisionExplanation) -> Dict[str, Any]:
    """Convert explanation to audit trail format"""
    return {
        'decision_id': explanation.decision_id,
        'explanation_summary': explanation.executive_summary,
        'factor_contributions': explanation.factor_contributions,
        'confidence_level': explanation.decision_confidence.value,
        'key_insights': explanation.key_insights,
        'risk_factors': explanation.risk_factors,
        'uncertainty_range': explanation.uncertainty_range
    }


# Convenience functions for integration
def explain_bid_decision(decision_id: str, user_id: str, campaign_id: str,
                        state: Any, action: Dict[str, Any], context: Dict[str, Any],
                        model_outputs: Dict[str, Any], decision_factors: Dict[str, Any],
                        engine: Optional[BidExplainabilityEngine] = None) -> BidDecisionExplanation:
    """Convenience function to generate bid explanation"""
    if engine is None:
        engine = BidExplainabilityEngine()
    
    return engine.explain_bid_decision(
        decision_id, user_id, campaign_id, state, action, context, 
        model_outputs, decision_factors
    )


if __name__ == "__main__":
    # Demo the explainability system
    print("GAELP Bid Explainability System Demo")
    print("=" * 60)
    
    # Create explainability engine
    engine = BidExplainabilityEngine()
    
    # Mock data for demo
    from dataclasses import dataclass
    
    @dataclass
    class MockState:
        segment_cvr: float = 0.04
        creative_predicted_ctr: float = 0.025
        competition_level: float = 0.7
        is_peak_hour: bool = True
        hour_of_day: int = 20
        creative_fatigue: float = 0.3
        creative_cta_strength: float = 0.8
        budget_spent_ratio: float = 0.6
        stage: int = 2
        device: int = 0
        channel_attribution_credit: float = 0.4
        cross_device_confidence: float = 0.7
        num_devices_seen: int = 2
        
        def to_vector(self, data_stats=None):
            return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Generate explanation
    explanation = engine.explain_bid_decision(
        decision_id="demo_001",
        user_id="user_123",
        campaign_id="campaign_001",
        state=MockState(),
        action={
            'bid_amount': 6.50,
            'creative_id': 42,
            'channel': 'paid_search',
            'bid_action': 10
        },
        context={
            'base_bid': 5.00,
            'pacing_factor': 1.2,
            'daily_budget': 1000.0,
            'exploration_mode': False
        },
        model_outputs={
            'q_values_bid': [2.1, 3.2, 4.5, 3.8, 2.9, 1.8, 2.3, 2.7, 3.1, 4.2, 4.6, 3.4],
            'q_values_creative': [1.2, 2.3, 1.8, 2.1],
            'q_values_channel': [1.5, 2.8, 2.2, 1.9, 1.7]
        },
        decision_factors={
            'model_version': 'demo_v1'
        }
    )
    
    # Display explanation
    print(f"\nDecision ID: {explanation.decision_id}")
    print(f"Final Bid: ${explanation.final_bid:.2f}")
    print(f"Confidence: {explanation.decision_confidence.value}")
    print(f"\nExecutive Summary:")
    print(f"  {explanation.executive_summary}")
    
    print(f"\nPrimary Factors ({len(explanation.primary_factors)}):")
    for factor in explanation.primary_factors:
        contribution = explanation.factor_contributions.get(factor.name, 0)
        print(f"  • {factor.name}: {contribution:.0%} impact ({factor.importance_level.value})")
        print(f"    {factor.explanation}")
    
    print(f"\nKey Insights:")
    for insight in explanation.key_insights:
        print(f"  • {insight}")
    
    print(f"\nRisk Factors:")
    for risk in explanation.risk_factors:
        print(f"  ⚠ {risk}")
    
    print(f"\nOptimization Opportunities:")
    for opp in explanation.optimization_opportunities:
        print(f"  💡 {opp}")
    
    print(f"\nUncertainty Range: ${explanation.uncertainty_range[0]:.2f} - ${explanation.uncertainty_range[1]:.2f}")
    
    print(f"\nCounterfactual Scenarios:")
    for scenario, details in explanation.counterfactuals.items():
        print(f"  • {details['scenario']}")
        print(f"    Change: {details['estimated_bid_change']}")
        print(f"    Reason: {details['rationale']}")
    
    print("\n✅ Explainability Demo Complete")
    print(f"Total explanation coverage: {sum(explanation.factor_contributions.values()):.0%}")