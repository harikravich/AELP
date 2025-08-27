from AuctionAllocation import AllocationMechanism
from Bidder import Bidder

import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import sys
import os

# Add path to access competitive intelligence
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
try:
    from competitive_intel import CompetitiveIntelligence, AuctionOutcome
except ImportError:
    CompetitiveIntelligence = None
    AuctionOutcome = None
    print("Warning: CompetitiveIntelligence not available")

from BidderAllocation import OracleAllocator
from Models import sigmoid

class Auction:
    ''' Base class for auctions with competitive intelligence integration '''
    def __init__(self, rng, allocation, agents, agent2items, agents2item_values, max_slots, embedding_size, embedding_var, obs_embedding_size, num_participants_per_round, enable_competitive_intel=True):
        self.rng = rng
        self.allocation = allocation
        self.agents = agents
        self.max_slots = max_slots
        self.revenue = .0

        self.agent2items = agent2items
        self.agents2item_values = agents2item_values

        self.embedding_size = embedding_size
        self.embedding_var = embedding_var

        self.obs_embedding_size = obs_embedding_size

        self.num_participants_per_round = num_participants_per_round
        
        # Competitive Intelligence Integration
        self.enable_competitive_intel = enable_competitive_intel and CompetitiveIntelligence is not None
        if self.enable_competitive_intel:
            self.competitive_intel = CompetitiveIntelligence(lookback_days=30)
        else:
            self.competitive_intel = None
            
        # Track auction outcomes for each agent
        self.agent_outcomes = {agent.name: [] for agent in agents}
        self.auction_count = 0
        
        # Safety and budget limits
        self.max_bid_multiplier = 3.0  # Maximum bid adjustment based on competitive pressure
        self.competitive_spend_limit = 1000.0  # Daily competitive spend limit

    def simulate_opportunity(self):
        self.auction_count += 1
        timestamp = datetime.now()
        
        # Sample the number of slots uniformly between [1, max_slots]
        num_slots = self.rng.integers(1, self.max_slots + 1)

        # Sample a true context vector
        true_context = np.concatenate((self.rng.normal(0, self.embedding_var, size=self.embedding_size), [1.0]))

        # Mask true context into observable context
        obs_context = np.concatenate((true_context[:self.obs_embedding_size], [1.0]))

        # At this point, the auctioneer solicits bids from
        # the list of bidders that might want to compete.
        original_bids = []
        adjusted_bids = []
        CTRs = []
        participating_agents_idx = self.rng.choice(len(self.agents), self.num_participants_per_round, replace=False)
        participating_agents = [self.agents[idx] for idx in participating_agents_idx]
        
        # Generate keyword for this auction (simplified)
        keyword = f"auction_keyword_{self.auction_count % 100}"
        
        for agent in participating_agents:
            # Get the original bid and the allocated item
            if isinstance(agent.allocator, OracleAllocator):
                original_bid, item = agent.bid(true_context)
            else:
                original_bid, item = agent.bid(obs_context)
                
            # Apply competitive intelligence if enabled
            if self.enable_competitive_intel and hasattr(agent, 'use_competitive_intel') and agent.use_competitive_intel:
                adjusted_bid = self._apply_competitive_intelligence(agent, original_bid, keyword, timestamp)
            else:
                adjusted_bid = original_bid
                
            original_bids.append(original_bid)
            adjusted_bids.append(adjusted_bid)
            
            # Compute the true CTRs for items in this agent's catalogue
            true_CTR = sigmoid(true_context @ self.agent2items[agent.name].T)
            agent.logs[-1].set_true_CTR(np.max(true_CTR * self.agents2item_values[agent.name]), true_CTR[item])
            CTRs.append(true_CTR[item])
            
        bids = np.array(adjusted_bids)
        CTRs = np.array(CTRs)

        # Now we have bids, we need to somehow allocate slots
        # "second_prices" tell us how much lower the winner could have gone without changing the outcome
        winners, prices, second_prices = self.allocation.allocate(bids, num_slots)

        # Bidders only obtain value when they get their outcome
        # Either P(view), P(click | view, ad), P(conversion | click, view, ad)
        # For now, look at P(click | ad) * P(view)
        outcomes = self.rng.binomial(1, CTRs[winners])

        # Let bidders know what they're being charged for and record outcomes
        for slot_id, (winner, price, second_price, outcome) in enumerate(zip(winners, prices, second_prices, outcomes)):
            for agent_id, agent in enumerate(participating_agents):
                if agent_id == winner:
                    agent.charge(price, second_price, bool(outcome))
                    
                    # Record auction outcome for competitive intelligence
                    if self.enable_competitive_intel:
                        self._record_auction_outcome(
                            agent, keyword, adjusted_bids[agent_id], slot_id + 1, 
                            price, timestamp, len(participating_agents)
                        )
                else:
                    agent.set_price(price)
                    
                    # Record loss for competitive intelligence
                    if self.enable_competitive_intel:
                        self._record_auction_outcome(
                            agent, keyword, adjusted_bids[agent_id], None, 
                            None, timestamp, len(participating_agents)
                        )
                        
            self.revenue += price
            
        # Update competitive intelligence patterns after auction
        if self.enable_competitive_intel:
            self.competitive_intel.track_patterns("market_aggregate")

    def _apply_competitive_intelligence(self, agent, original_bid: float, keyword: str, timestamp: datetime) -> float:
        """Apply competitive intelligence to adjust bid based on competition"""
        try:
            if not self.competitive_intel or not hasattr(agent, 'quality_score'):
                return original_bid
                
            # Create a dummy outcome for estimation
            dummy_outcome = AuctionOutcome(
                timestamp=timestamp,
                keyword=keyword,
                our_bid=original_bid,
                position=None,
                cost=None,
                competitor_count=self.num_participants_per_round - 1,
                quality_score=getattr(agent, 'quality_score', 7.0),
                daypart=timestamp.hour,
                day_of_week=timestamp.weekday(),
                device_type="desktop",
                location="default"
            )
            
            # Estimate competitor bid for position 1
            competitor_bid, confidence = self.competitive_intel.estimate_competitor_bid(dummy_outcome, position=1)
            
            if competitor_bid > 0 and confidence > 0.2:  # Only adjust if we have reasonable confidence
                # Predict market response to our bid increase
                response = self.competitive_intel.predict_response(
                    original_bid * 1.2,  # 20% increase scenario
                    keyword,
                    timestamp,
                    "bid_increase"
                )
                
                # Calculate competitive pressure multiplier
                competitive_pressure = response.get('competitor_responses', {}).get('escalation_probability', 0.5)
                market_saturation = 1.0 if response.get('market_impact', {}).get('market_saturation_risk', False) else 0.7
                
                # Determine bid adjustment based on competitive intelligence
                if competitor_bid > original_bid:
                    # We need to bid higher to compete
                    bid_multiplier = min(
                        (competitor_bid / original_bid) * 1.05,  # 5% above competitor
                        self.max_bid_multiplier  # Respect safety limits
                    )
                    
                    # Reduce multiplier if high competitive pressure (avoid bidding wars)
                    if competitive_pressure > 0.7:
                        bid_multiplier = min(bid_multiplier, 1.3)
                    
                    # Apply market saturation factor
                    bid_multiplier *= market_saturation
                    
                    adjusted_bid = original_bid * bid_multiplier
                    
                    # Safety check: don't exceed spend limits
                    if adjusted_bid > original_bid and hasattr(agent, 'daily_spend'):
                        remaining_budget = self.competitive_spend_limit - getattr(agent, 'daily_spend', 0)
                        if adjusted_bid - original_bid > remaining_budget * 0.1:  # Max 10% of remaining budget
                            adjusted_bid = original_bid + remaining_budget * 0.1
                    
                    return max(original_bid, adjusted_bid)
                else:
                    # Our bid is competitive, maybe reduce it slightly
                    return original_bid * 0.95
            
            return original_bid
            
        except Exception as e:
            print(f"Warning: Competitive intelligence adjustment failed: {e}")
            return original_bid
    
    def _record_auction_outcome(self, agent, keyword: str, bid: float, position: Optional[int], 
                               cost: Optional[float], timestamp: datetime, competitor_count: int):
        """Record auction outcome for competitive intelligence"""
        try:
            outcome = AuctionOutcome(
                timestamp=timestamp,
                keyword=keyword,
                our_bid=bid,
                position=position,
                cost=cost,
                competitor_count=competitor_count,
                quality_score=getattr(agent, 'quality_score', 7.0),
                daypart=timestamp.hour,
                day_of_week=timestamp.weekday(),
                device_type="desktop",
                location="default"
            )
            
            self.competitive_intel.record_auction_outcome(outcome)
            self.agent_outcomes[agent.name].append(outcome)
            
        except Exception as e:
            print(f"Warning: Failed to record auction outcome: {e}")
    
    def get_competitive_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of competitive intelligence insights"""
        if not self.enable_competitive_intel:
            return {"error": "Competitive intelligence not enabled"}
            
        try:
            summary = self.competitive_intel.get_market_intelligence_summary()
            summary["agent_outcomes"] = {
                agent_name: len(outcomes) 
                for agent_name, outcomes in self.agent_outcomes.items()
            }
            summary["total_auctions"] = self.auction_count
            return summary
        except Exception as e:
            return {"error": f"Failed to generate summary: {e}"}
    
    def estimate_competitor_response(self, agent_name: str, planned_bid: float, keyword: str) -> Dict[str, Any]:
        """Estimate how competitors will respond to our planned bid"""
        if not self.enable_competitive_intel:
            return {"error": "Competitive intelligence not enabled"}
            
        try:
            timestamp = datetime.now() + timedelta(hours=1)  # Future bid
            response = self.competitive_intel.predict_response(
                planned_bid, keyword, timestamp, "bid_increase"
            )
            return response
        except Exception as e:
            return {"error": f"Failed to estimate response: {e}"}
    
    def clear_revenue(self):
        self.revenue = 0.0
