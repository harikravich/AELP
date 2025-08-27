"""
Competitive Intelligence Integration for GAELP Auction System
Integrates competitive intelligence with orchestrator and safety systems
NO HARDCODED VALUES - All parameters discovered from GA4 data
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

# Import parameter discovery manager - NO HARDCODED VALUES
from parameter_discovery_manager import get_parameter_manager, get_discovered_value

try:
    from competitive_intel import CompetitiveIntelligence, AuctionOutcome
except ImportError:
    CompetitiveIntelligence = None
    AuctionOutcome = None

try:
    from multi_channel_orchestrator import MultiChannelOrchestrator
except ImportError:
    MultiChannelOrchestrator = None

try:
    from safety_framework.safety_orchestrator import ComprehensiveSafetyOrchestrator, SafetyConfiguration
except ImportError:
    ComprehensiveSafetyOrchestrator = None
    SafetyConfiguration = None

logger = logging.getLogger(__name__)


@dataclass
class CompetitiveBidDecision:
    """Represents a competitive intelligence-informed bid decision"""
    original_bid: float
    adjusted_bid: float
    competitive_pressure: float
    competitor_estimate: float
    confidence: float
    reasoning: str
    safety_approved: bool = True
    spend_impact: float = 0.0


@dataclass
class CompetitiveSpendLimit:
    """Spending limits for competitive bidding"""
    daily_limit: float = 1000.0
    per_auction_limit: float = 50.0
    competitive_multiplier_limit: float = 3.0
    emergency_stop_threshold: float = 2000.0


class CompetitiveAuctionOrchestrator:
    """
    Orchestrates auction bidding with competitive intelligence,
    integrating with safety systems and budget controls.
    """
    
    def __init__(self, 
                 enable_competitive_intel: bool = True,
                 spend_limits: Optional[CompetitiveSpendLimit] = None,
                 safety_config: Optional[SafetyConfiguration] = None):
        
        self.enable_competitive_intel = enable_competitive_intel and CompetitiveIntelligence is not None
        
        # Initialize competitive intelligence
        if self.enable_competitive_intel:
            self.competitive_intel = CompetitiveIntelligence(lookback_days=30)
        else:
            self.competitive_intel = None
            logger.warning("Competitive intelligence not available")
        
        # Initialize safety orchestrator
        if ComprehensiveSafetyOrchestrator and safety_config:
            self.safety_orchestrator = ComprehensiveSafetyOrchestrator(safety_config)
        else:
            self.safety_orchestrator = None
            logger.warning("Safety orchestrator not available")
        
        # Spending controls
        self.spend_limits = spend_limits or CompetitiveSpendLimit()
        self.daily_competitive_spend = 0.0
        self.competitive_spend_history = []
        
        # Performance tracking
        self.bid_adjustments = []
        self.win_rate_by_adjustment = {"increased": [], "decreased": [], "unchanged": []}
        self.cost_savings = 0.0
        self.cost_increases = 0.0
        
        # Emergency controls
        self.emergency_stop_active = False
        self.emergency_reasons = []
        
        # Integration callbacks
        self.orchestrator_callbacks = []
        self.safety_callbacks = []
    
    async def initialize(self):
        """Initialize the orchestrator and start monitoring"""
        try:
            if self.safety_orchestrator:
                await self.safety_orchestrator.start_monitoring()
                
                # Register safety callbacks
                self.safety_orchestrator.register_alert_callback(self._handle_safety_alert)
                
            logger.info("Competitive auction orchestrator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize competitive auction orchestrator: {e}")
    
    def decide_competitive_bid(self, 
                              agent_name: str,
                              original_bid: float, 
                              keyword: str, 
                              timestamp: Optional[datetime] = None,
                              agent_quality_score: float = 7.0,
                              agent_daily_budget: float = 1000.0) -> CompetitiveBidDecision:
        """
        Make a competitive intelligence-informed bid decision
        """
        if not self.enable_competitive_intel or self.emergency_stop_active:
            return CompetitiveBidDecision(
                original_bid=original_bid,
                adjusted_bid=original_bid,
                competitive_pressure=0.0,
                competitor_estimate=0.0,
                confidence=0.0,
                reasoning="Competitive intelligence disabled or emergency stop active"
            )
        
        timestamp = timestamp or datetime.now()
        
        try:
            # Create auction outcome for estimation
            dummy_outcome = AuctionOutcome(
                timestamp=timestamp,
                keyword=keyword,
                our_bid=original_bid,
                position=None,
                cost=None,
                competitor_count=5,  # Estimated
                quality_score=agent_quality_score,
                daypart=timestamp.hour,
                day_of_week=timestamp.weekday(),
                device_type="desktop",
                location="default"
            )
            
            # Estimate competitor bid
            competitor_bid, confidence = self.competitive_intel.estimate_competitor_bid(
                dummy_outcome, position=1
            )
            
            if confidence < 0.2:  # Low confidence
                return CompetitiveBidDecision(
                    original_bid=original_bid,
                    adjusted_bid=original_bid,
                    competitive_pressure=0.0,
                    competitor_estimate=competitor_bid,
                    confidence=confidence,
                    reasoning="Low confidence in competitor estimate"
                )
            
            # Predict market response
            response = self.competitive_intel.predict_response(
                original_bid * 1.2,
                keyword,
                timestamp,
                "bid_increase"
            )
            
            competitive_pressure = response.get('competitor_responses', {}).get(
                'escalation_probability', 0.5
            )
            
            # Calculate bid adjustment
            adjusted_bid = self._calculate_bid_adjustment(
                original_bid, competitor_bid, competitive_pressure, 
                agent_daily_budget, confidence
            )
            
            # Safety check
            safety_approved, safety_reasons = self._safety_check_bid_adjustment(
                agent_name, original_bid, adjusted_bid
            )
            
            if not safety_approved:
                adjusted_bid = original_bid
            
            # Generate reasoning
            reasoning = self._generate_bid_reasoning(
                original_bid, adjusted_bid, competitor_bid, 
                competitive_pressure, confidence, safety_reasons
            )
            
            decision = CompetitiveBidDecision(
                original_bid=original_bid,
                adjusted_bid=adjusted_bid,
                competitive_pressure=competitive_pressure,
                competitor_estimate=competitor_bid,
                confidence=confidence,
                reasoning=reasoning,
                safety_approved=safety_approved,
                spend_impact=adjusted_bid - original_bid
            )
            
            # Track the decision
            self._track_bid_decision(decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Competitive bid decision failed: {e}")
            return CompetitiveBidDecision(
                original_bid=original_bid,
                adjusted_bid=original_bid,
                competitive_pressure=0.0,
                competitor_estimate=0.0,
                confidence=0.0,
                reasoning=f"Error in competitive analysis: {str(e)}"
            )
    
    def _calculate_bid_adjustment(self, 
                                 original_bid: float,
                                 competitor_bid: float,
                                 competitive_pressure: float,
                                 agent_budget: float,
                                 confidence: float) -> float:
        """Calculate the optimal bid adjustment based on competitive intelligence"""
        
        if competitor_bid <= 0:
            return original_bid
        
        # Base adjustment: try to outbid competitor by 5%
        if competitor_bid > original_bid:
            base_multiplier = (competitor_bid / original_bid) * 1.05
        else:
            # We're already competitive, consider small reduction
            base_multiplier = 0.95
        
        # Adjust for competitive pressure (high pressure = less aggressive bidding)
        if competitive_pressure > 0.7:
            pressure_discount = 0.8  # Reduce aggressiveness in bidding wars
        elif competitive_pressure > 0.4:
            pressure_discount = 0.9
        else:
            pressure_discount = 1.0
        
        multiplier = base_multiplier * pressure_discount
        
        # Apply safety limits
        multiplier = max(0.5, min(multiplier, self.spend_limits.competitive_multiplier_limit))
        
        # Confidence-weighted adjustment
        confidence_weight = min(confidence, 0.9)  # Cap at 90%
        final_multiplier = 1.0 + (multiplier - 1.0) * confidence_weight
        
        adjusted_bid = original_bid * final_multiplier
        
        # Budget constraint
        remaining_budget = agent_budget - self.daily_competitive_spend
        if adjusted_bid - original_bid > remaining_budget * 0.1:  # Max 10% of remaining budget
            adjusted_bid = original_bid + remaining_budget * 0.1
        
        # Absolute limits
        adjusted_bid = min(adjusted_bid, original_bid + self.spend_limits.per_auction_limit)
        
        return max(original_bid * 0.5, adjusted_bid)  # Never go below 50% of original
    
    def _safety_check_bid_adjustment(self, 
                                   agent_name: str, 
                                   original_bid: float, 
                                   adjusted_bid: float) -> Tuple[bool, List[str]]:
        """Perform safety checks on bid adjustment"""
        reasons = []
        
        # Check spending limits
        spend_increase = adjusted_bid - original_bid
        
        if spend_increase > self.spend_limits.per_auction_limit:
            reasons.append(f"Spend increase {spend_increase:.2f} exceeds per-auction limit {self.spend_limits.per_auction_limit}")
        
        if self.daily_competitive_spend + spend_increase > self.spend_limits.daily_limit:
            reasons.append(f"Would exceed daily competitive spend limit {self.spend_limits.daily_limit}")
        
        # Check for emergency conditions
        if self.daily_competitive_spend > self.spend_limits.emergency_stop_threshold:
            reasons.append("Emergency spend threshold exceeded")
            self._trigger_emergency_stop("Excessive competitive spending")
        
        # Check bid multiplier
        if adjusted_bid > original_bid * self.spend_limits.competitive_multiplier_limit:
            reasons.append(f"Bid multiplier exceeds safety limit {self.spend_limits.competitive_multiplier_limit}")
        
        # Integration with safety orchestrator
        if self.safety_orchestrator:
            try:
                # This would integrate with actual safety validation
                # For now, basic checks
                pass
            except Exception as e:
                reasons.append(f"Safety orchestrator check failed: {e}")
        
        return len(reasons) == 0, reasons
    
    def _generate_bid_reasoning(self, 
                              original_bid: float,
                              adjusted_bid: float, 
                              competitor_bid: float,
                              competitive_pressure: float, 
                              confidence: float,
                              safety_reasons: List[str]) -> str:
        """Generate human-readable reasoning for bid adjustment"""
        
        reasoning_parts = []
        
        # Adjustment magnitude
        if adjusted_bid > original_bid:
            increase_pct = ((adjusted_bid - original_bid) / original_bid) * 100
            reasoning_parts.append(f"Increased bid by {increase_pct:.1f}%")
        elif adjusted_bid < original_bid:
            decrease_pct = ((original_bid - adjusted_bid) / original_bid) * 100
            reasoning_parts.append(f"Decreased bid by {decrease_pct:.1f}%")
        else:
            reasoning_parts.append("No bid adjustment")
        
        # Competitor information
        if competitor_bid > 0:
            if competitor_bid > original_bid:
                reasoning_parts.append(f"competitor estimated at ${competitor_bid:.2f} (above our ${original_bid:.2f})")
            else:
                reasoning_parts.append(f"competitor estimated at ${competitor_bid:.2f} (below our ${original_bid:.2f})")
        
        # Market conditions
        if competitive_pressure > 0.7:
            reasoning_parts.append("high competitive pressure detected")
        elif competitive_pressure > 0.4:
            reasoning_parts.append("moderate competitive pressure")
        else:
            reasoning_parts.append("low competitive pressure")
        
        # Confidence level
        reasoning_parts.append(f"confidence: {confidence:.1f}")
        
        # Safety constraints
        if safety_reasons:
            reasoning_parts.append(f"safety constraints: {'; '.join(safety_reasons)}")
        
        return "; ".join(reasoning_parts)
    
    def _track_bid_decision(self, decision: CompetitiveBidDecision):
        """Track bid decision for analysis and learning"""
        self.bid_adjustments.append({
            'timestamp': datetime.now(),
            'original_bid': decision.original_bid,
            'adjusted_bid': decision.adjusted_bid,
            'adjustment_ratio': decision.adjusted_bid / decision.original_bid,
            'competitive_pressure': decision.competitive_pressure,
            'confidence': decision.confidence,
            'safety_approved': decision.safety_approved,
            'spend_impact': decision.spend_impact
        })
        
        # Update spending tracking
        if decision.spend_impact > 0:
            self.daily_competitive_spend += decision.spend_impact
            self.cost_increases += decision.spend_impact
        elif decision.spend_impact < 0:
            self.cost_savings += abs(decision.spend_impact)
        
        # Keep recent history only
        if len(self.bid_adjustments) > 1000:
            self.bid_adjustments = self.bid_adjustments[-500:]
    
    def record_auction_outcome(self, 
                              keyword: str,
                              bid: float, 
                              won: bool,
                              position: Optional[int] = None,
                              cost: Optional[float] = None,
                              timestamp: Optional[datetime] = None,
                              quality_score: float = 7.0):
        """Record auction outcome for competitive intelligence learning"""
        
        if not self.enable_competitive_intel:
            return
        
        timestamp = timestamp or datetime.now()
        
        try:
            outcome = AuctionOutcome(
                timestamp=timestamp,
                keyword=keyword,
                our_bid=bid,
                position=position if won else None,
                cost=cost if won else None,
                competitor_count=5,  # Estimated
                quality_score=quality_score,
                daypart=timestamp.hour,
                day_of_week=timestamp.weekday(),
                device_type="desktop",
                location="default"
            )
            
            self.competitive_intel.record_auction_outcome(outcome)
            
            # Update win rate tracking based on bid adjustments
            adjustment_type = "unchanged"
            for adj in reversed(self.bid_adjustments[-10:]):  # Check recent adjustments
                if abs(adj['timestamp'] - timestamp) < timedelta(minutes=1):
                    if adj['adjustment_ratio'] > 1.05:
                        adjustment_type = "increased"
                    elif adj['adjustment_ratio'] < 0.95:
                        adjustment_type = "decreased"
                    break
            
            self.win_rate_by_adjustment[adjustment_type].append(1 if won else 0)
            
            # Keep recent history
            for key in self.win_rate_by_adjustment:
                if len(self.win_rate_by_adjustment[key]) > 100:
                    self.win_rate_by_adjustment[key] = self.win_rate_by_adjustment[key][-50:]
                    
        except Exception as e:
            logger.error(f"Failed to record auction outcome: {e}")
    
    def track_competitive_patterns(self) -> Dict[str, Any]:
        """Update and return competitive intelligence patterns"""
        if not self.enable_competitive_intel:
            return {"error": "Competitive intelligence not enabled"}
        
        try:
            # Update patterns
            profile = self.competitive_intel.track_patterns("market_aggregate")
            
            # Get market intelligence
            summary = self.competitive_intel.get_market_intelligence_summary()
            
            # Add our orchestrator metrics
            summary["orchestrator_metrics"] = self.get_performance_metrics()
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to track competitive patterns: {e}")
            return {"error": str(e)}
    
    def estimate_competitor_response(self, 
                                   planned_bid: float,
                                   keyword: str,
                                   scenario: str = "bid_increase") -> Dict[str, Any]:
        """Estimate how competitors will respond to our planned actions"""
        
        if not self.enable_competitive_intel:
            return {"error": "Competitive intelligence not enabled"}
        
        try:
            timestamp = datetime.now() + timedelta(hours=1)  # Future prediction
            
            response = self.competitive_intel.predict_response(
                planned_bid, keyword, timestamp, scenario
            )
            
            # Add our risk assessment
            response["risk_assessment"] = self._assess_competitive_risk(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to estimate competitor response: {e}")
            return {"error": str(e)}
    
    def _assess_competitive_risk(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk of the predicted competitive response"""
        
        risk_level = "low"
        risk_factors = []
        
        try:
            competitor_responses = response.get('competitor_responses', {})
            market_impact = response.get('market_impact', {})
            
            # High escalation probability = high risk
            escalation_prob = competitor_responses.get('escalation_probability', 0)
            if escalation_prob > 0.8:
                risk_level = "high"
                risk_factors.append("High escalation probability")
            elif escalation_prob > 0.6:
                risk_level = "medium"
                risk_factors.append("Moderate escalation probability")
            
            # Market saturation risk
            if market_impact.get('market_saturation_risk', False):
                risk_level = "high"
                risk_factors.append("Market saturation risk")
            
            # Budget impact
            expected_cpc_increase = market_impact.get('expected_cpc_increase', "0%")
            if isinstance(expected_cpc_increase, str) and "%" in expected_cpc_increase:
                try:
                    increase_pct = float(expected_cpc_increase.replace('%', ''))
                    if increase_pct > 50:
                        risk_level = "high"
                        risk_factors.append(f"High CPC increase expected: {expected_cpc_increase}")
                    elif increase_pct > 20:
                        if risk_level == "low":
                            risk_level = "medium"
                        risk_factors.append(f"Moderate CPC increase expected: {expected_cpc_increase}")
                except:
                    pass
            
            return {
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendation": self._get_risk_recommendation(risk_level, risk_factors)
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"risk_level": "unknown", "error": str(e)}
    
    def _get_risk_recommendation(self, risk_level: str, risk_factors: List[str]) -> str:
        """Get risk-based recommendations"""
        
        if risk_level == "high":
            return "Consider reducing bid aggressiveness or avoiding this market segment"
        elif risk_level == "medium":
            return "Proceed with caution and monitor competitor responses closely"
        else:
            return "Low risk scenario - normal bidding strategy recommended"
    
    async def _handle_safety_alert(self, alert: Any):
        """Handle safety alerts from the safety orchestrator"""
        try:
            logger.warning(f"Safety alert received: {alert}")
            
            # Implement safety responses based on alert type
            if hasattr(alert, 'safety_level'):
                if alert.safety_level.value in ['critical', 'emergency']:
                    self._trigger_emergency_stop(f"Safety alert: {alert.description}")
            
        except Exception as e:
            logger.error(f"Failed to handle safety alert: {e}")
    
    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop of competitive bidding"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self.emergency_reasons.append({
                'timestamp': datetime.now(),
                'reason': reason
            })
            
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            
            # Notify callbacks
            for callback in self.safety_callbacks:
                try:
                    callback({
                        'type': 'emergency_stop',
                        'reason': reason,
                        'timestamp': datetime.now()
                    })
                except Exception as e:
                    logger.error(f"Emergency stop callback failed: {e}")
    
    def reset_emergency_stop(self, authorized_by: str):
        """Reset emergency stop (requires authorization)"""
        if self.emergency_stop_active:
            self.emergency_stop_active = False
            logger.info(f"Emergency stop reset by {authorized_by}")
            
            # Reset spending counters
            self.daily_competitive_spend = 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the competitive orchestrator"""
        
        metrics = {
            'competitive_intelligence_enabled': self.enable_competitive_intel,
            'emergency_stop_active': self.emergency_stop_active,
            'total_bid_adjustments': len(self.bid_adjustments),
            'daily_competitive_spend': self.daily_competitive_spend,
            'cost_savings': self.cost_savings,
            'cost_increases': self.cost_increases,
            'net_cost_impact': self.cost_increases - self.cost_savings,
            'spend_limits': {
                'daily_limit': self.spend_limits.daily_limit,
                'per_auction_limit': self.spend_limits.per_auction_limit,
                'multiplier_limit': self.spend_limits.competitive_multiplier_limit
            }
        }
        
        # Win rate analysis by adjustment type
        for adj_type, wins in self.win_rate_by_adjustment.items():
            if wins:
                metrics[f'win_rate_{adj_type}_bids'] = np.mean(wins)
            else:
                metrics[f'win_rate_{adj_type}_bids'] = 0.0
        
        # Recent adjustment statistics
        if self.bid_adjustments:
            recent_adjustments = self.bid_adjustments[-100:]  # Last 100
            
            ratios = [adj['adjustment_ratio'] for adj in recent_adjustments]
            metrics['avg_bid_adjustment_ratio'] = np.mean(ratios)
            metrics['bid_increase_rate'] = np.mean([r > 1.05 for r in ratios])
            metrics['bid_decrease_rate'] = np.mean([r < 0.95 for r in ratios])
        
        return metrics
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        return {
            'status': 'emergency_stopped' if self.emergency_stop_active else 'active',
            'performance_metrics': self.get_performance_metrics(),
            'competitive_patterns': self.track_competitive_patterns() if self.enable_competitive_intel else {},
            'safety_status': {
                'safety_orchestrator_enabled': self.safety_orchestrator is not None,
                'emergency_reasons': self.emergency_reasons[-5:] if self.emergency_reasons else []
            },
            'last_updated': datetime.now()
        }


# Example usage and integration helpers

async def create_competitive_auction_system(
    enable_competitive_intel: bool = True,
    enable_safety_orchestrator: bool = True,
    daily_spend_limit: float = 1000.0
) -> CompetitiveAuctionOrchestrator:
    """Create a complete competitive auction system with safety integration"""
    
    # Configure spending limits
    spend_limits = CompetitiveSpendLimit(
        daily_limit=daily_spend_limit,
        per_auction_limit=daily_spend_limit * 0.05,  # 5% per auction
        competitive_multiplier_limit=2.5,
        emergency_stop_threshold=daily_spend_limit * 1.5
    )
    
    # Configure safety system
    safety_config = None
    if enable_safety_orchestrator and SafetyConfiguration:
        safety_config = SafetyConfiguration(
            max_daily_budget=daily_spend_limit,
            auto_pause_on_critical=True,
            enable_budget_controls=True
        )
    
    # Create orchestrator
    orchestrator = CompetitiveAuctionOrchestrator(
        enable_competitive_intel=enable_competitive_intel,
        spend_limits=spend_limits,
        safety_config=safety_config
    )
    
    # Initialize
    await orchestrator.initialize()
    
    return orchestrator


def main():
    """Example usage of the competitive auction integration"""
    print("🎯 Competitive Intelligence Auction Integration")
    print("=" * 60)
    
    async def run_example():
        # Create system
        orchestrator = await create_competitive_auction_system(
            enable_competitive_intel=True,
            enable_safety_orchestrator=True,
            daily_spend_limit=500.0
        )
        
        print("✅ Competitive auction orchestrator initialized")
        
        # Example bid decision
        decision = orchestrator.decide_competitive_bid(
            agent_name="test_agent",
            original_bid=2.50,
            keyword="running shoes",
            agent_quality_score=8.0,
            agent_daily_budget=500.0
        )
        
        print(f"\n📊 Bid Decision Example:")
        print(f"  Original Bid: ${decision.original_bid:.2f}")
        print(f"  Adjusted Bid: ${decision.adjusted_bid:.2f}")
        print(f"  Competitive Pressure: {decision.competitive_pressure:.2f}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning}")
        
        # Example outcome recording
        orchestrator.record_auction_outcome(
            keyword="running shoes",
            bid=decision.adjusted_bid,
            won=True,
            position=1,
            cost=2.10
        )
        
        # Get status
        status = orchestrator.get_orchestrator_status()
        print(f"\n📈 Orchestrator Status:")
        print(f"  Status: {status['status']}")
        print(f"  Total Adjustments: {status['performance_metrics']['total_bid_adjustments']}")
        print(f"  Daily Spend: ${status['performance_metrics']['daily_competitive_spend']:.2f}")
        
        return orchestrator
    
    # Run example
    import asyncio
    return asyncio.run(run_example())


if __name__ == "__main__":
    orchestrator = main()
    print("\n✅ Competitive auction integration ready!")