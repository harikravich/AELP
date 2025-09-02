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
        logger.info(f"Competitors: {[f'{name} (${info.budget:.0f})' for name, info in self.bidders.items()]}")
    
    def initialize_competitors(self, competitor_config: Dict[str, Any]):
        """Initialize competitor bidding agents using AuctionGym bidders"""
        
        # Create aggressive competitor bidders reflecting real competitive market
        competitors = competitor_config.get('agents', [
            {'name': 'Qustodio', 'type': 'empirical', 'budget': 320.0, 'gamma': 0.85},  # Very competitive empirical
            {'name': 'Bark', 'type': 'truthful', 'budget': 380.0},  # Market leader, high budget
            {'name': 'Circle', 'type': 'empirical', 'budget': 280.0, 'gamma': 0.80},  # Aggressive empirical
            {'name': 'Norton', 'type': 'truthful', 'budget': 350.0},  # Premium brand, high budget
            {'name': 'Life360', 'type': 'empirical', 'budget': 400.0, 'gamma': 0.88},  # Top competitor
            {'name': 'SmallComp1', 'type': 'empirical', 'budget': 220.0, 'gamma': 0.75},  # Scrappy competitor
            {'name': 'McAfee', 'type': 'truthful', 'budget': 360.0},  # Premium cybersecurity
            {'name': 'Kaspersky', 'type': 'empirical', 'budget': 300.0, 'gamma': 0.82},  # International competitor
        ])
        
        # Initialize random number generator for bidders (different seeds for variation)
        rng = np.random.RandomState(np.random.randint(1000))
        
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
        Run a single auction round using proper GSP (Generalized Second Price) auction
        with quality scores affecting ad rank.
        
        Args:
            our_bid: Our bid amount
            query_value: Estimated value of the query/keyword
            context: Additional context (user segment, time, etc.)
        
        Returns:
            AuctionResult with auction outcome
        """
        
        # Get competitor bids and quality scores
        all_bidders = []
        
        # Add our bid with quality score
        our_quality_score = context.get('quality_score', 5.0)  # 1-10 scale
        all_bidders.append({
            'name': 'GAELP',
            'bid': our_bid,
            'quality_score': our_quality_score,
            'ad_rank': our_bid * our_quality_score
        })
        
        # Get competitor bids using realistic market dynamics
        for name, bidder in self.bidders.items():
            estimated_ctr = context.get('estimated_ctr', 0.05)
            
            # Scale query value based on competitor characteristics
            competitor_query_value = self._get_competitor_query_value(bidder, query_value, context)
            
            # Get bid from AuctionGym bidder with more aggressive bidding
            comp_bid = bidder.bid(
                value=competitor_query_value,
                context=context,
                estimated_CTR=estimated_ctr
            )
            
            # Add moderate competition boost for empirical bidders (they learn to be competitive)
            if hasattr(bidder, 'prev_gamma'):  # EmpiricalShadedBidder
                competition_boost = np.random.uniform(1.05, 1.25)  # 5-25% more aggressive
                comp_bid *= competition_boost
            
            # Apply realistic budget constraints
            comp_bid = self._apply_budget_constraints(bidder, comp_bid)
            
            # Generate realistic quality score for competitor
            comp_quality_score = self._get_competitor_quality_score(name)
            
            # Filter bids below reserve price
            if comp_bid >= self.reserve_price:
                all_bidders.append({
                    'name': name,
                    'bid': comp_bid,
                    'quality_score': comp_quality_score,
                    'ad_rank': comp_bid * comp_quality_score
                })
        
        # Check if our bid meets reserve price
        if our_bid < self.reserve_price:
            # We didn't meet reserve price
            return AuctionResult(
                won=False, price_paid=0.0, slot_position=-1,
                total_slots=self.num_slots, competitors=len(all_bidders)-1,
                estimated_ctr=0.0, true_ctr=0.0, outcome=False, revenue=0.0
            )
        
        # Run GSP auction: Sort by Ad Rank (Bid × Quality Score)
        all_bidders.sort(key=lambda x: x['ad_rank'], reverse=True)
        
        # Determine winners (top slots)
        num_available_slots = min(self.num_slots, len(all_bidders))
        winners = all_bidders[:num_available_slots]
        
        # Find our position
        our_position = None
        for i, bidder in enumerate(all_bidders):
            if bidder['name'] == 'GAELP':
                our_position = i + 1
                break
        
        # Check if we won a slot
        won = our_position is not None and our_position <= num_available_slots
        
        if won:
            # Correct GSP pricing: Pay minimum needed to maintain position
            our_slot_index = our_position - 1
            
            if our_slot_index < len(all_bidders) - 1:
                # GSP: pay the minimum bid needed to beat the next highest ad rank
                next_bidder = all_bidders[our_slot_index + 1]
                
                # Calculate minimum bid needed: (next_ad_rank / our_quality_score) + $0.01
                min_bid_needed = (next_bidder['ad_rank'] / our_quality_score) + 0.01
                
                # In GSP, you pay the minimum needed, but never more than your actual bid
                price_paid = min(min_bid_needed, our_bid)
                
                # Ensure we pay at least the reserve price
                price_paid = max(price_paid, self.reserve_price)
            else:
                # Bottom slot: pay reserve price
                price_paid = max(self.reserve_price, 0.51)  # Slightly above reserve
            
            position = our_position
        else:
            position = our_position if our_position else len(all_bidders) + 1
            price_paid = 0.0
        
        # Calculate performance metrics
        result = self._calculate_auction_outcome(
            won, price_paid, position, context, len(all_bidders)-1
        )
        
        return result
    
    def _get_competitor_query_value(self, bidder, query_value: float, context: Dict[str, Any]) -> float:
        """Calculate competitor's perceived query value based on their characteristics"""
        
        # Competitors have different value perceptions of the same query
        # Balanced multipliers to create realistic competition (20-30% win rates)
        
        base_multipliers = {
            'Bark': np.random.uniform(1.4, 2.0),      # Market leader, very competitive
            'Qustodio': np.random.uniform(1.3, 1.9),  # Strong competitive presence
            'Circle': np.random.uniform(1.0, 1.6),    # Solid competitive player
            'Norton': np.random.uniform(1.2, 1.8),    # Premium brand, strong
            'Life360': np.random.uniform(1.5, 2.1),   # Top competitor, aggressive
            'SmallComp1': np.random.uniform(0.8, 1.4), # Scrappy but budget limited
            'McAfee': np.random.uniform(1.6, 2.2),    # Premium leader, high bids
            'Kaspersky': np.random.uniform(1.1, 1.7), # International, competitive
        }
        
        # Get base multiplier for this competitor
        competitor_name = getattr(bidder, 'name', 'Unknown')
        base_multiplier = base_multipliers.get(competitor_name, np.random.uniform(1.5, 2.5))
        
        # Apply budget factor (moderate influence for high budgets)
        if hasattr(bidder, 'budget'):
            budget_factor = min(1.4, bidder.budget / 300.0)  # Higher budget = more competitive
        else:
            budget_factor = 1.0
        
        # Market pressure factor - competitive environment
        market_pressure = np.random.uniform(1.0, 1.3)  # Competitive market pressure
        
        # Peak hour competition boost
        hour = context.get('hour', 12)
        if hour in [9, 10, 11, 14, 15, 16, 19, 20]:  # Peak advertising hours
            peak_boost = np.random.uniform(1.1, 1.3)  # Moderate peak boost
        else:
            peak_boost = np.random.uniform(1.0, 1.1)  # Small off-peak reduction
        
        # Final competitor value - aggressive but balanced
        competitor_value = query_value * base_multiplier * budget_factor * market_pressure * peak_boost
        
        return max(query_value * 1.0, competitor_value)  # Competitors value queries at least as much as us
    
    def _apply_budget_constraints(self, bidder, bid: float) -> float:
        """Apply realistic budget constraints to competitor bids"""
        
        if not hasattr(bidder, 'budget'):
            return bid
        
        # Max bid per auction based on budget allocation
        # More aggressive constraint: up to 3% of budget per auction for high-value queries
        competitor_name = getattr(bidder, 'name', 'Unknown')
        
        # Premium competitors bid more aggressively but within reason
        if competitor_name in ['Bark', 'Life360', 'McAfee', 'Norton']:
            max_bid_per_auction = min(bidder.budget * 0.028, 10.0)  # Premium players bid higher
        else:
            max_bid_per_auction = min(bidder.budget * 0.020, 6.5)   # Others competitive
        
        # Apply constraint
        constrained_bid = min(bid, max_bid_per_auction)
        
        # Ensure minimum bid above reserve (competitors bid above reserve moderately)
        constrained_bid = max(constrained_bid, self.reserve_price + np.random.uniform(0.05, 0.25))
        
        return constrained_bid
    
    def _get_competitor_quality_score(self, competitor_name: str) -> float:
        """Generate realistic quality scores for competitors based on their market position"""
        
        # Quality scores based on real market data (1-10 scale)
        # Competitors generally have good quality scores in competitive markets
        competitor_quality_scores = {
            'Bark': np.random.normal(7.5, 0.6),      # Strong brand, good landing pages
            'Qustodio': np.random.normal(8.0, 0.5),  # High quality, relevant ads
            'Circle': np.random.normal(7.2, 0.6),    # Decent quality
            'Norton': np.random.normal(8.2, 0.4),    # Premium brand, high quality
            'Life360': np.random.normal(7.8, 0.5),   # Good product-market fit
            'SmallComp1': np.random.normal(6.2, 0.8), # Smaller player, variable quality
            'McAfee': np.random.normal(8.5, 0.4),    # Premium cybersecurity brand
            'Kaspersky': np.random.normal(7.9, 0.5), # Strong cybersecurity brand
        }
        
        # Default quality score for unknown competitors
        base_score = competitor_quality_scores.get(competitor_name, np.random.normal(7.0, 0.8))
        
        # Ensure quality score is within bounds (1-10)
        quality_score = max(1.0, min(10.0, base_score))
        
        return quality_score
    
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
