#!/usr/bin/env python3
"""
AuctionGym Integration for GAELP - FIXED VERSION
This module uses Amazon's AuctionGym with proper second-price mechanics and competitive bidding.
"""

import sys
import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Add auction-gym to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'auction-gym', 'src'))

# Import AuctionGym components
from Auction import Auction
from Agent import Agent
from Bidder import TruthfulBidder, EmpiricalShadedBidder
from AuctionAllocation import AllocationMechanism, SecondPrice, FirstPrice
from BidderAllocation import OracleAllocator
from Models import sigmoid

logger = logging.getLogger(__name__)

@dataclass
class AuctionResult:
    """Result from a single auction round"""
    won: bool
    price_paid: float
    slot_position: int
    total_slots: int
    competitors: int
    estimated_ctr: float
    true_ctr: float
    outcome: bool  # Did the user click?
    revenue: float

class AuctionGymWrapper:
    """
    Wrapper around Amazon's AuctionGym for ad auction simulation.
    Implements proper second-price auctions with realistic competitive bidding.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with AuctionGym components"""
        config = config or {}
        
        # Configure auction type
        self.auction_type = config.get('auction_type', 'second_price')
        self.num_slots = config.get('num_slots', 4)  # Google typically shows 4 ads
        self.reserve_price = config.get('reserve_price', 0.50)
        
        # Handle competitor count from config
        competitor_config = config.get('competitors', {})
        if isinstance(competitor_config, dict) and 'count' in competitor_config:
            self.num_competitors = competitor_config['count']
        else:
            self.num_competitors = competitor_config.get('agents', 6) if isinstance(competitor_config, dict) else 6
        
        # Initialize real AuctionGym allocation mechanism
        if self.auction_type == 'second_price':
            self.allocation_mechanism = SecondPrice()
        else:
            self.allocation_mechanism = FirstPrice()
        
        # Initialize bidders
        self.bidders = {}
        self.initialize_competitors(config.get('competitors', {}))
        
        logger.info(f"AuctionGym initialized with {self.auction_type} auction, {self.num_slots} slots, {len(self.bidders)} competitors")
    
    def initialize_competitors(self, competitor_config: Dict[str, Any]):
        """Initialize competitor bidding agents using AuctionGym bidders"""
        
        # Create realistic competitor bidders with higher budgets and competitive strategies
        competitors = competitor_config.get('agents', [
            {'name': 'Qustodio', 'type': 'empirical', 'budget': 250.0, 'gamma': 0.65},  # Competitive empirical bidder
            {'name': 'Bark', 'type': 'truthful', 'budget': 300.0},  # High budget truthful bidder
            {'name': 'Circle', 'type': 'empirical', 'budget': 280.0, 'gamma': 0.70},  # Moderate empirical bidder
            {'name': 'Norton', 'type': 'truthful', 'budget': 220.0},  # Medium budget truthful
            {'name': 'Life360', 'type': 'empirical', 'budget': 320.0, 'gamma': 0.60},  # Very competitive
            {'name': 'SmallComp1', 'type': 'empirical', 'budget': 150.0, 'gamma': 0.50},  # Aggressive small player
        ])
        
        # Initialize random number generator for bidders
        rng = np.random.RandomState(42)
        
        for comp in competitors:
            if comp['type'] == 'empirical':
                # EmpiricalShadedBidder with varied gamma (shading factor)
                bidder = EmpiricalShadedBidder(
                    rng=rng,
                    gamma_sigma=0.15,  # More variation in shading for competitiveness
                    init_gamma=comp.get('gamma', 0.75)  # Initial shading factor
                )
            else:
                # TruthfulBidder - bids true value
                bidder = TruthfulBidder(rng=rng)
            
            # Store budget and name separately
            bidder.budget = comp['budget']
            bidder.name = comp['name']
            
            self.bidders[comp['name']] = bidder
            logger.info(f"Initialized {comp['type']} bidder: {comp['name']} with ${comp['budget']} budget")
    
    def run_auction(self, our_bid: float, query_value: float, 
                   context: Dict[str, Any]) -> AuctionResult:
        """
        Run a single auction round using AuctionGym
        
        Args:
            our_bid: Our bid amount
            query_value: Estimated value of the query/keyword
            context: Additional context (user segment, time, etc.)
        
        Returns:
            AuctionResult with auction outcome
        """
        
        # Get competitor bids using realistic market dynamics
        competitor_bids = []
        for name, bidder in self.bidders.items():
            # Get estimated CTR from context
            estimated_ctr = context.get('estimated_ctr', 0.05)
            
            # Scale query value based on competitor characteristics to create realistic competition
            competitor_query_value = self._get_competitor_query_value(bidder, query_value, context)
            
            # Get bid from AuctionGym bidder
            comp_bid = bidder.bid(
                value=competitor_query_value,
                context=context,
                estimated_CTR=estimated_ctr
            )
            
            # Apply realistic budget constraints
            comp_bid = self._apply_budget_constraints(bidder, comp_bid)
            
            competitor_bids.append(comp_bid)
        
        # Run auction through AuctionGym allocation mechanism
        all_bids = np.array([our_bid] + competitor_bids)
        bidder_names = ['GAELP'] + list(self.bidders.keys())
        
        # Filter bids below reserve price
        valid_indices = np.where(all_bids >= self.reserve_price)[0]
        if len(valid_indices) == 0:
            # No valid bids
            return AuctionResult(
                won=False, price_paid=0.0, slot_position=-1,
                total_slots=self.num_slots, competitors=len(competitor_bids),
                estimated_ctr=0.0, true_ctr=0.0, outcome=False, revenue=0.0
            )
        
        # Use AuctionGym allocation mechanism for proper second-price auction
        valid_bids = all_bids[valid_indices]
        num_slots = min(self.num_slots, len(valid_indices))
        
        winners, prices, _ = self.allocation_mechanism.allocate(
            bids=valid_bids,
            num_slots=num_slots
        )
        
        # Map back to original indices
        actual_winners = valid_indices[winners]
        
        # Check if we won (we're always index 0)
        our_index = 0
        won = our_index in actual_winners
        
        if won:
            win_position = np.where(actual_winners == our_index)[0][0]
            position = win_position + 1
            price_paid = prices[win_position]
        else:
            position = len(all_bids) + 1
            price_paid = 0.0
        
        # Calculate performance metrics
        result = self._calculate_auction_outcome(
            won, price_paid, position, context, len(competitor_bids)
        )
        
        return result
    
    def _get_competitor_query_value(self, bidder, query_value: float, context: Dict[str, Any]) -> float:
        """Calculate competitor's perceived query value based on their characteristics"""
        
        # Base value is the same for all
        competitor_value = query_value
        
        if hasattr(bidder, 'budget'):
            # Higher budget competitors can afford to see higher value
            # Scale relative to average budget of $200
            budget_multiplier = min(1.8, bidder.budget / 200.0)
            
            # Market competitiveness factor (2.0 = 2x more competitive than baseline)
            market_factor = 1.8
            
            competitor_value = query_value * budget_multiplier * market_factor
        
        # Add some randomness for realistic variation
        noise_factor = np.random.normal(1.0, 0.1)  # ±10% variation
        competitor_value *= max(0.5, noise_factor)  # Don't go below 50% of base
        
        return competitor_value
    
    def _apply_budget_constraints(self, bidder, bid: float) -> float:
        """Apply realistic budget constraints to competitor bids"""
        
        if not hasattr(bidder, 'budget'):
            return bid
        
        # Max bid per auction based on budget allocation
        # Allow up to 2% of annual budget per auction for competitive markets
        max_bid_per_auction = min(bidder.budget * 0.02, 8.0)
        
        # Apply constraint
        constrained_bid = min(bid, max_bid_per_auction)
        
        # Ensure minimum bid above reserve
        constrained_bid = max(constrained_bid, self.reserve_price)
        
        return constrained_bid
    
    def _calculate_auction_outcome(self, won: bool, price_paid: float, 
                                 position: int, context: Dict[str, Any], 
                                 num_competitors: int) -> AuctionResult:
        """Calculate auction outcome including CTR, clicks, and revenue"""
        
        # Google Ads position-based CTRs (realistic values)
        position_ctrs = {
            1: 0.055,  # First position
            2: 0.028,  # Second position
            3: 0.018,  # Third position
            4: 0.012   # Fourth position
        }
        
        estimated_ctr = position_ctrs.get(position, 0.003) if won else 0.0
        
        # Apply quality score multiplier
        quality_score = context.get('quality_score', 1.0)
        true_ctr = estimated_ctr * quality_score
        
        # Simulate click outcome
        clicked = np.random.random() < true_ctr if won else False
        
        # Calculate revenue if conversion happens
        revenue = 0.0
        if clicked:
            cvr = context.get('conversion_rate', 0.02)
            if np.random.random() < cvr:
                revenue = context.get('customer_ltv', 199.98)  # Aura customer LTV
        
        return AuctionResult(
            won=won,
            price_paid=price_paid,
            slot_position=position if won else -1,
            total_slots=self.num_slots,
            competitors=num_competitors,
            estimated_ctr=estimated_ctr,
            true_ctr=true_ctr,
            outcome=clicked,
            revenue=revenue
        )
    
    def get_market_stats(self) -> Dict[str, Any]:
        """Get market statistics"""
        return {
            'total_competitors': len(self.bidders),
            'auction_type': self.auction_type,
            'num_slots': self.num_slots,
            'reserve_price': self.reserve_price,
            'competitor_names': list(self.bidders.keys())
        }
    
    def get_competitor_insights(self) -> Dict[str, Any]:
        """Get insights about competitor bidding patterns"""
        insights = {}
        
        for name, bidder in self.bidders.items():
            insights[name] = {
                'name': name,
                'type': type(bidder).__name__,
                'budget': getattr(bidder, 'budget', 0),
                'gamma': getattr(bidder, 'prev_gamma', None) if hasattr(bidder, 'prev_gamma') else None
            }
        
        return insights

# Ensure AuctionGym is properly loaded
print("✅ AuctionGym Fixed Integration loaded - Real second-price auction mechanics!")
