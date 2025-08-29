#!/usr/bin/env python3
"""
GAELP Dynamic Budget Optimizer
Real-time budget optimization for $1000/day across Google, Facebook, TikTok.
NO STATIC ALLOCATIONS - Pure performance-driven optimization.

Features:
- Dynamic channel allocation based on marginal ROAS
- Dayparting with crisis/decision time multipliers
- iOS premium bidding (20-30%)
- Real-time pacing algorithms
- Performance-based reallocation
- Channel minimum/maximum constraints
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import logging
import asyncio
import json
from enum import Enum
import math

logger = logging.getLogger(__name__)


class GAELPChannel(Enum):
    """GAELP-specific marketing channels"""
    GOOGLE_SEARCH = "google_search"
    GOOGLE_DISPLAY = "google_display"
    FACEBOOK_FEED = "facebook_feed"
    FACEBOOK_STORIES = "facebook_stories"
    TIKTOK_FEED = "tiktok_feed"
    TIKTOK_SPARK = "tiktok_spark"


class DeviceType(Enum):
    """Device targeting types"""
    IOS = "ios"
    ANDROID = "android"
    DESKTOP = "desktop"


@dataclass
class ChannelConstraints:
    """Channel-specific constraints and parameters"""
    min_daily_budget: Decimal
    max_daily_budget: Decimal
    learning_phase_budget: Decimal  # Minimum for algorithm learning
    scaling_limit: Decimal  # Point of diminishing returns
    target_roas: float
    current_roas: float
    marginal_roas: float  # ROAS at next dollar
    ios_multiplier: float  # iOS bid premium
    priority_score: float  # Channel priority (1-10)


@dataclass
class DaypartMultiplier:
    """Hour-specific bid multipliers"""
    hour: int
    multiplier: float
    reason: str
    conversion_probability: float
    competition_level: float


@dataclass
class BidDecision:
    """Real-time bidding decision"""
    campaign_id: str
    channel: GAELPChannel
    device: DeviceType
    hour: int
    base_bid: Decimal
    daypart_multiplier: float
    device_multiplier: float
    final_bid: Decimal
    expected_roas: float
    spend_approved: bool
    reason: str


@dataclass
class PerformanceMetrics:
    """Real-time performance tracking"""
    channel: GAELPChannel
    spend: Decimal
    impressions: int
    clicks: int
    conversions: int
    revenue: Decimal
    roas: float
    cpa: Decimal
    efficiency_score: float  # Normalized performance metric
    last_updated: datetime


class MarginalROASCalculator:
    """Calculate marginal ROAS for budget allocation optimization"""
    
    def __init__(self):
        self.historical_data = {}
        self.efficiency_curves = {}
    
    def calculate_marginal_roas(self, channel: GAELPChannel, current_spend: Decimal, 
                               performance_data: List[PerformanceMetrics]) -> float:
        """
        Calculate marginal ROAS at next dollar of spend.
        Uses efficiency curve analysis to determine diminishing returns.
        """
        try:
            if not performance_data:
                return self._get_default_marginal_roas(channel)
            
            # Sort by spend level
            sorted_data = sorted(performance_data, key=lambda x: x.spend)
            
            if len(sorted_data) < 3:
                return sorted_data[-1].roas if sorted_data else self._get_default_marginal_roas(channel)
            
            # Calculate efficiency curve
            spend_points = [float(d.spend) for d in sorted_data]
            roas_points = [d.roas for d in sorted_data]
            
            # Fit polynomial curve to find diminishing returns
            coefficients = np.polyfit(spend_points, roas_points, min(3, len(spend_points)-1))
            
            # Calculate derivative (marginal ROAS) at current spend level
            derivative_coeffs = [i * coefficients[-(i+2)] for i in range(1, len(coefficients))]
            derivative_coeffs.reverse()
            
            current_spend_float = float(current_spend)
            marginal_roas = sum(coeff * (current_spend_float ** i) for i, coeff in enumerate(derivative_coeffs))
            
            # Apply channel-specific constraints
            return max(0.1, min(10.0, marginal_roas))
            
        except Exception as e:
            logger.error(f"Error calculating marginal ROAS: {e}")
            return self._get_default_marginal_roas(channel)
    
    def _get_default_marginal_roas(self, channel: GAELPChannel) -> float:
        """Default marginal ROAS estimates by channel"""
        defaults = {
            GAELPChannel.GOOGLE_SEARCH: 3.5,
            GAELPChannel.GOOGLE_DISPLAY: 2.1,
            GAELPChannel.FACEBOOK_FEED: 2.8,
            GAELPChannel.FACEBOOK_STORIES: 2.3,
            GAELPChannel.TIKTOK_FEED: 1.9,
            GAELPChannel.TIKTOK_SPARK: 2.1
        }
        return defaults.get(channel, 2.0)


class DaypartingEngine:
    """Advanced dayparting with behavioral health specific patterns"""
    
    def __init__(self):
        self.daypart_config = self._initialize_daypart_multipliers()
        self.timezone_adjustments = {}
    
    def _initialize_daypart_multipliers(self) -> Dict[int, DaypartMultiplier]:
        """Initialize hour-specific multipliers based on behavioral health patterns"""
        return {
            # Late night crisis searches (high intent, low competition)
            0: DaypartMultiplier(0, 1.4, "crisis_parents", 0.35, 0.2),
            1: DaypartMultiplier(1, 1.4, "crisis_parents", 0.35, 0.2),
            2: DaypartMultiplier(2, 1.4, "crisis_parents", 0.35, 0.2),
            3: DaypartMultiplier(3, 1.3, "late_night_worry", 0.28, 0.3),
            
            # Early morning (low activity)
            4: DaypartMultiplier(4, 0.7, "low_activity", 0.15, 0.1),
            5: DaypartMultiplier(5, 0.7, "low_activity", 0.15, 0.1),
            6: DaypartMultiplier(6, 0.8, "early_morning", 0.18, 0.2),
            7: DaypartMultiplier(7, 0.9, "morning_prep", 0.22, 0.4),
            8: DaypartMultiplier(8, 1.0, "school_prep", 0.25, 0.6),
            
            # Work hours (research time)
            9: DaypartMultiplier(9, 1.1, "research_phase", 0.28, 0.7),
            10: DaypartMultiplier(10, 1.2, "morning_research", 0.30, 0.8),
            11: DaypartMultiplier(11, 1.1, "pre_lunch", 0.28, 0.8),
            
            # Lunch break (mobile heavy)
            12: DaypartMultiplier(12, 1.2, "mobile_browsing", 0.32, 0.9),
            13: DaypartMultiplier(13, 1.2, "lunch_break", 0.32, 0.9),
            14: DaypartMultiplier(14, 1.1, "afternoon_start", 0.28, 0.8),
            
            # After school (parent concern time)
            15: DaypartMultiplier(15, 1.3, "after_school", 0.38, 1.0),
            16: DaypartMultiplier(16, 1.3, "after_school", 0.38, 1.0),
            17: DaypartMultiplier(17, 1.3, "after_school", 0.38, 1.0),
            18: DaypartMultiplier(18, 1.2, "dinner_prep", 0.35, 0.9),
            
            # Evening (family discussion time - PEAK)
            19: DaypartMultiplier(19, 1.5, "decision_time", 0.45, 1.2),
            20: DaypartMultiplier(20, 1.5, "decision_time", 0.45, 1.2),
            21: DaypartMultiplier(21, 1.5, "decision_time", 0.45, 1.2),
            22: DaypartMultiplier(22, 1.2, "final_research", 0.35, 0.8),
            23: DaypartMultiplier(23, 1.1, "late_evening", 0.30, 0.6)
        }
    
    def get_multiplier(self, hour: int, device: DeviceType = DeviceType.DESKTOP) -> float:
        """Get daypart multiplier for specific hour and device"""
        base_multiplier = self.daypart_config.get(hour, DaypartMultiplier(hour, 1.0, "default", 0.25, 0.5))
        
        # Apply device-specific adjustments
        if device == DeviceType.IOS:
            # iOS users more likely to convert during mobile-heavy hours
            if hour in [12, 13, 15, 16, 17]:
                return base_multiplier.multiplier * 1.15
            return base_multiplier.multiplier * 1.05
        elif device == DeviceType.ANDROID:
            return base_multiplier.multiplier * 1.0
        else:  # Desktop
            # Desktop stronger during research hours
            if hour in [9, 10, 11, 19, 20, 21]:
                return base_multiplier.multiplier * 1.08
            return base_multiplier.multiplier * 0.95
    
    def get_expected_conversion_rate(self, hour: int) -> float:
        """Get expected conversion rate for the hour"""
        return self.daypart_config.get(hour, DaypartMultiplier(hour, 1.0, "default", 0.25, 0.5)).conversion_probability


class ChannelOptimizer:
    """Optimize budget allocation across channels based on performance"""
    
    def __init__(self):
        self.channel_constraints = self._initialize_channel_constraints()
        self.marginal_calculator = MarginalROASCalculator()
        
    def _initialize_channel_constraints(self) -> Dict[GAELPChannel, ChannelConstraints]:
        """Initialize channel constraints based on GAELP requirements"""
        return {
            GAELPChannel.GOOGLE_SEARCH: ChannelConstraints(
                min_daily_budget=Decimal('150'),  # Need volume for behavioral health keywords
                max_daily_budget=Decimal('600'),  # Diminishing returns above this
                learning_phase_budget=Decimal('100'),
                scaling_limit=Decimal('400'),
                target_roas=3.2,
                current_roas=3.2,
                marginal_roas=3.5,
                ios_multiplier=1.25,  # 25% premium for iOS searches
                priority_score=9.0
            ),
            GAELPChannel.GOOGLE_DISPLAY: ChannelConstraints(
                min_daily_budget=Decimal('50'),
                max_daily_budget=Decimal('200'),
                learning_phase_budget=Decimal('30'),
                scaling_limit=Decimal('150'),
                target_roas=2.1,
                current_roas=2.1,
                marginal_roas=2.3,
                ios_multiplier=1.15,
                priority_score=6.0
            ),
            GAELPChannel.FACEBOOK_FEED: ChannelConstraints(
                min_daily_budget=Decimal('200'),  # Facebook needs higher minimums for learning
                max_daily_budget=Decimal('400'),
                learning_phase_budget=Decimal('150'),
                scaling_limit=Decimal('300'),
                target_roas=2.8,
                current_roas=2.1,
                marginal_roas=2.4,
                ios_multiplier=1.30,  # Higher iOS premium on Facebook
                priority_score=8.0
            ),
            GAELPChannel.FACEBOOK_STORIES: ChannelConstraints(
                min_daily_budget=Decimal('75'),
                max_daily_budget=Decimal('200'),
                learning_phase_budget=Decimal('50'),
                scaling_limit=Decimal('150'),
                target_roas=2.3,
                current_roas=1.9,
                marginal_roas=2.1,
                ios_multiplier=1.35,  # Stories very iOS heavy
                priority_score=7.0
            ),
            GAELPChannel.TIKTOK_FEED: ChannelConstraints(
                min_daily_budget=Decimal('100'),
                max_daily_budget=Decimal('300'),
                learning_phase_budget=Decimal('75'),
                scaling_limit=Decimal('200'),
                target_roas=1.9,
                current_roas=1.8,
                marginal_roas=1.9,
                ios_multiplier=1.20,  # Moderate iOS premium
                priority_score=6.5
            ),
            GAELPChannel.TIKTOK_SPARK: ChannelConstraints(
                min_daily_budget=Decimal('50'),
                max_daily_budget=Decimal('150'),
                learning_phase_budget=Decimal('40'),
                scaling_limit=Decimal('100'),
                target_roas=2.1,
                current_roas=1.8,
                marginal_roas=2.0,
                ios_multiplier=1.15,
                priority_score=5.5
            )
        }
    
    def calculate_optimal_allocation(self, daily_budget: Decimal, 
                                   current_performance: Dict[GAELPChannel, PerformanceMetrics]) -> Dict[GAELPChannel, Decimal]:
        """
        Calculate optimal budget allocation using marginal utility optimization.
        NO STATIC PERCENTAGES - Pure performance-driven.
        """
        try:
            logger.info(f"Calculating optimal allocation for ${daily_budget}")
            
            # Step 1: Ensure minimum budgets are met
            allocation = {}
            remaining_budget = daily_budget
            
            for channel, constraints in self.channel_constraints.items():
                allocation[channel] = constraints.min_daily_budget
                remaining_budget -= constraints.min_daily_budget
                logger.debug(f"Allocated minimum ${constraints.min_daily_budget} to {channel.value}")
            
            if remaining_budget < 0:
                raise ValueError(f"Daily budget ${daily_budget} insufficient for minimum allocations")
            
            # Step 2: Allocate remaining budget using marginal ROAS optimization
            while remaining_budget > Decimal('10'):  # Continue until <$10 left
                best_channel = None
                best_marginal_roas = 0
                
                for channel, constraints in self.channel_constraints.items():
                    # Skip if at maximum
                    if allocation[channel] >= constraints.max_daily_budget:
                        continue
                    
                    # Calculate marginal ROAS for next $10 increment
                    performance_data = [current_performance.get(channel)] if channel in current_performance else []
                    marginal_roas = self.marginal_calculator.calculate_marginal_roas(
                        channel, allocation[channel], performance_data
                    )
                    
                    # Apply diminishing returns
                    if allocation[channel] > constraints.scaling_limit:
                        diminishing_factor = 1 - float((allocation[channel] - constraints.scaling_limit) / 
                                                 (constraints.max_daily_budget - constraints.scaling_limit))
                        marginal_roas *= max(0.1, diminishing_factor)
                    
                    if marginal_roas > best_marginal_roas:
                        best_marginal_roas = marginal_roas
                        best_channel = channel
                
                if best_channel is None:
                    logger.warning("No channel can accept more budget")
                    break
                
                # Allocate $10 increment to best performing channel
                increment = min(Decimal('10'), remaining_budget, 
                              self.channel_constraints[best_channel].max_daily_budget - allocation[best_channel])
                allocation[best_channel] += increment
                remaining_budget -= increment
                
                logger.debug(f"Allocated ${increment} to {best_channel.value} (marginal ROAS: {best_marginal_roas:.2f})")
            
            # Step 3: Add any remaining budget to highest performing channel
            total_allocated = sum(allocation.values())
            remaining_budget = daily_budget - total_allocated
            
            if remaining_budget > Decimal('0.01'):
                # Add remainder to Google Search (highest priority)
                allocation[GAELPChannel.GOOGLE_SEARCH] += remaining_budget
                total_allocated = daily_budget
            
            if abs(total_allocated - daily_budget) > Decimal('1'):
                logger.warning(f"Allocation mismatch: ${total_allocated} vs ${daily_budget}")
            
            logger.info(f"Optimal allocation: {[(k.value, float(v)) for k, v in allocation.items()]}")
            return allocation
            
        except Exception as e:
            logger.error(f"Error calculating optimal allocation: {e}")
            # No fallback - fail loudly if optimization fails
            logger.critical("OPTIMIZATION FAILED - No fallback allowed!")
            raise RuntimeError("Budget optimization failed. Fix the optimizer instead of using fallbacks.")
    
    def _removed_fallback_allocation(self, daily_budget: Decimal) -> Dict[GAELPChannel, Decimal]:
        """REMOVED - No fallback allocations allowed in production"""
        raise RuntimeError("Fallback allocation not allowed. Fix the optimization failure.")
        
        for channel, constraints in self.channel_constraints.items():
            proportional_share = remaining * Decimal(str(constraints.priority_score / total_priority))
            allocation[channel] = constraints.min_daily_budget + proportional_share
            
        return allocation


class BudgetPacer:
    """Real-time budget pacing to prevent early exhaustion"""
    
    def __init__(self, daily_budget: Decimal):
        self.daily_budget = daily_budget
        self.spent_today = Decimal('0')
        self.hourly_spend = {}
        self.pacing_multiplier = 1.0
        self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
    def calculate_pacing_multiplier(self) -> float:
        """Calculate bid adjustment based on spend vs time progress"""
        current_time = datetime.now()
        elapsed_hours = (current_time - self.start_time).total_seconds() / 3600
        
        if elapsed_hours <= 0:
            return 1.0
            
        # Calculate ideal vs actual spend ratio
        time_progress = elapsed_hours / 24.0
        spend_progress = float(self.spent_today / self.daily_budget)
        
        if spend_progress == 0:
            # No spend yet - normal pacing
            return 1.0
        
        pace_ratio = spend_progress / time_progress
        
        # Adjust pacing multiplier
        if pace_ratio > 1.5:  # Spending too fast
            return 0.6
        elif pace_ratio > 1.2:
            return 0.8
        elif pace_ratio < 0.7:  # Spending too slow
            return 1.4
        elif pace_ratio < 0.9:
            return 1.2
        else:
            return 1.0
    
    def can_spend(self, amount: Decimal, hour: int) -> bool:
        """Check if spend amount is within pacing limits"""
        # Check daily limit
        if self.spent_today + amount > self.daily_budget * Decimal('0.95'):
            return False
        
        # Check hourly pacing (max 15% of daily budget per hour)
        max_hourly = self.daily_budget * Decimal('0.15')
        hourly_spent = self.hourly_spend.get(hour, Decimal('0'))
        
        if hourly_spent + amount > max_hourly:
            return False
        
        return True
    
    def record_spend(self, amount: Decimal, hour: int):
        """Record spend and update pacing"""
        self.spent_today += amount
        if hour not in self.hourly_spend:
            self.hourly_spend[hour] = Decimal('0')
        self.hourly_spend[hour] += amount
        self.pacing_multiplier = self.calculate_pacing_multiplier()


class GAELPBudgetOptimizer:
    """
    Main GAELP Budget Optimization Engine
    Real-time, dynamic, performance-driven budget allocation
    """
    
    def __init__(self, daily_budget: Decimal = Decimal('1000')):
        self.daily_budget = daily_budget
        self.channel_optimizer = ChannelOptimizer()
        self.dayparting_engine = DaypartingEngine()
        self.budget_pacer = BudgetPacer(daily_budget)
        self.current_allocations = {}
        self.performance_history = {}
        self.bid_decisions = []
        
        logger.info(f"GAELP Budget Optimizer initialized with ${daily_budget} daily budget")
    
    def get_current_allocation(self) -> Dict[GAELPChannel, Decimal]:
        """Get current optimal channel allocation"""
        return self.channel_optimizer.calculate_optimal_allocation(
            self.daily_budget, 
            self.performance_history
        )
    
    def make_bid_decision(self, campaign_id: str, channel: GAELPChannel, 
                         device: DeviceType, base_bid: Decimal, hour: int = None) -> BidDecision:
        """
        Make real-time bidding decision with all optimizations applied
        """
        current_hour = hour if hour is not None else datetime.now().hour
        
        # Get channel allocation
        allocations = self.get_current_allocation()
        channel_budget = allocations.get(channel, Decimal('0'))
        
        # Check if we have budget remaining for this channel
        channel_spent = sum(
            decision.final_bid for decision in self.bid_decisions
            if decision.channel == channel and decision.spend_approved
        )
        
        if channel_spent >= channel_budget:
            return BidDecision(
                campaign_id=campaign_id,
                channel=channel,
                device=device,
                hour=current_hour,
                base_bid=base_bid,
                daypart_multiplier=1.0,
                device_multiplier=1.0,
                final_bid=Decimal('0'),
                expected_roas=0.0,
                spend_approved=False,
                reason="Channel budget exhausted"
            )
        
        # Calculate dayparting multiplier
        daypart_multiplier = self.dayparting_engine.get_multiplier(current_hour, device)
        
        # Calculate device multiplier (iOS premium)
        constraints = self.channel_optimizer.channel_constraints[channel]
        if device == DeviceType.IOS:
            device_multiplier = constraints.ios_multiplier
        else:
            device_multiplier = 1.0
        
        # Apply pacing multiplier
        pacing_multiplier = self.budget_pacer.calculate_pacing_multiplier()
        
        # Calculate final bid
        final_bid = base_bid * Decimal(str(daypart_multiplier)) * Decimal(str(device_multiplier)) * Decimal(str(pacing_multiplier))
        
        # Check pacing constraints
        can_spend = self.budget_pacer.can_spend(final_bid, current_hour)
        
        # Calculate expected ROAS
        expected_conversion_rate = self.dayparting_engine.get_expected_conversion_rate(current_hour)
        expected_roas = constraints.current_roas * daypart_multiplier * (device_multiplier if device == DeviceType.IOS else 1.0)
        
        bid_decision = BidDecision(
            campaign_id=campaign_id,
            channel=channel,
            device=device,
            hour=current_hour,
            base_bid=base_bid,
            daypart_multiplier=daypart_multiplier,
            device_multiplier=device_multiplier,
            final_bid=final_bid if can_spend else Decimal('0'),
            expected_roas=expected_roas,
            spend_approved=can_spend,
            reason="Approved" if can_spend else "Pacing limit exceeded"
        )
        
        self.bid_decisions.append(bid_decision)
        
        if can_spend:
            self.budget_pacer.record_spend(final_bid, current_hour)
        
        return bid_decision
    
    def update_performance(self, channel: GAELPChannel, metrics: PerformanceMetrics):
        """Update channel performance data for optimization"""
        self.performance_history[channel] = metrics
        
        # Trigger reallocation if significant performance change
        if self._should_reallocate():
            self.reallocate_budget()
    
    def _should_reallocate(self) -> bool:
        """Determine if budget should be reallocated based on performance"""
        if len(self.performance_history) < 2:
            return False
        
        # Check for significant ROAS changes
        for channel, metrics in self.performance_history.items():
            constraints = self.channel_optimizer.channel_constraints[channel]
            if abs(metrics.roas - constraints.current_roas) > 0.5:  # 0.5 ROAS difference threshold
                return True
        
        return False
    
    def reallocate_budget(self):
        """Reallocate budget based on updated performance"""
        logger.info("Reallocating budget based on performance changes")
        
        # Update current ROAS in constraints
        for channel, metrics in self.performance_history.items():
            self.channel_optimizer.channel_constraints[channel].current_roas = metrics.roas
        
        # Recalculate optimal allocation
        self.current_allocations = self.get_current_allocation()
    
    def get_status_report(self) -> Dict:
        """Get comprehensive status report"""
        total_spend = sum(
            decision.final_bid for decision in self.bid_decisions
            if decision.spend_approved
        )
        
        channel_spend = {}
        for decision in self.bid_decisions:
            if decision.spend_approved:
                if decision.channel not in channel_spend:
                    channel_spend[decision.channel] = Decimal('0')
                channel_spend[decision.channel] += decision.final_bid
        
        ios_decisions = sum(1 for d in self.bid_decisions if d.device == DeviceType.IOS and d.spend_approved)
        total_decisions = sum(1 for d in self.bid_decisions if d.spend_approved)
        
        return {
            "daily_budget": float(self.daily_budget),
            "total_spend": float(total_spend),
            "budget_utilization": float(total_spend / self.daily_budget) if self.daily_budget > 0 else 0,
            "channel_allocations": {k.value: float(v) for k, v in self.get_current_allocation().items()},
            "channel_spend": {k.value: float(v) for k, v in channel_spend.items()},
            "pacing_multiplier": self.budget_pacer.pacing_multiplier,
            "total_bid_decisions": len(self.bid_decisions),
            "approved_decisions": total_decisions,
            "ios_decisions": ios_decisions,
            "ios_percentage": (ios_decisions / total_decisions * 100) if total_decisions > 0 else 0,
            "performance_data_points": len(self.performance_history)
        }


async def demo_gaelp_optimizer():
    """Demonstration of GAELP Budget Optimizer"""
    print("🚀 GAELP Dynamic Budget Optimizer Demo")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = GAELPBudgetOptimizer(Decimal('1000'))
    
    # Show initial allocation
    initial_allocation = optimizer.get_current_allocation()
    print(f"\n📊 Initial Optimal Allocation:")
    for channel, budget in initial_allocation.items():
        print(f"   {channel.value}: ${budget}")
    
    # Simulate bidding throughout the day
    print(f"\n⏰ Simulating 24-hour bidding with dayparting:")
    
    channels = [GAELPChannel.GOOGLE_SEARCH, GAELPChannel.FACEBOOK_FEED, GAELPChannel.TIKTOK_FEED]
    devices = [DeviceType.IOS, DeviceType.ANDROID, DeviceType.DESKTOP]
    
    hourly_results = []
    
    for hour in range(24):
        hour_decisions = []
        
        # Simulate 10 bid opportunities per hour
        for i in range(10):
            channel = np.random.choice(channels)
            device = np.random.choice(devices)
            base_bid = Decimal(str(np.random.uniform(2.0, 8.0)))
            
            decision = optimizer.make_bid_decision(
                f"campaign_{i}", channel, device, base_bid, hour
            )
            hour_decisions.append(decision)
        
        approved_decisions = [d for d in hour_decisions if d.spend_approved]
        total_spend = sum(d.final_bid for d in approved_decisions)
        avg_multiplier = np.mean([d.daypart_multiplier for d in hour_decisions])
        ios_decisions = sum(1 for d in approved_decisions if d.device == DeviceType.IOS)
        
        hourly_results.append({
            "hour": hour,
            "decisions": len(hour_decisions),
            "approved": len(approved_decisions),
            "spend": float(total_spend),
            "avg_multiplier": avg_multiplier,
            "ios_decisions": ios_decisions
        })
        
        # Show key hours
        if hour in [2, 9, 15, 19]:  # Crisis, research, after-school, decision time
            daypart = optimizer.dayparting_engine.daypart_config[hour]
            print(f"   Hour {hour:2d} ({daypart.reason:15s}): "
                  f"{avg_multiplier:.1f}x multiplier, "
                  f"${total_spend:6.2f} spend, "
                  f"{ios_decisions}/10 iOS decisions")
    
    # Show final status
    print(f"\n📈 Final Status Report:")
    status = optimizer.get_status_report()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: ${v:.2f}" if 'allocation' in key or 'spend' in key else f"      {k}: {v}")
        else:
            if 'budget' in key or 'spend' in key:
                print(f"   {key}: ${value:.2f}")
            elif 'percentage' in key or 'utilization' in key:
                print(f"   {key}: {value:.1f}%")
            else:
                print(f"   {key}: {value}")
    
    # Simulate performance update and reallocation
    print(f"\n🔄 Simulating Performance Update:")
    
    # Google Search performing better than expected
    google_metrics = PerformanceMetrics(
        channel=GAELPChannel.GOOGLE_SEARCH,
        spend=Decimal('400'),
        impressions=15000,
        clicks=750,
        conversions=65,
        revenue=Decimal('2600'),  # Higher than expected
        roas=6.5,  # Much higher than 3.2 target
        cpa=Decimal('40'),
        efficiency_score=0.9,
        last_updated=datetime.now()
    )
    
    optimizer.update_performance(GAELPChannel.GOOGLE_SEARCH, google_metrics)
    
    # Facebook underperforming
    facebook_metrics = PerformanceMetrics(
        channel=GAELPChannel.FACEBOOK_FEED,
        spend=Decimal('300'),
        impressions=25000,
        clicks=500,
        conversions=15,
        revenue=Decimal('450'),  # Lower than expected
        roas=1.5,  # Much lower than 2.8 target
        cpa=Decimal('100'),
        efficiency_score=0.3,
        last_updated=datetime.now()
    )
    
    optimizer.update_performance(GAELPChannel.FACEBOOK_FEED, facebook_metrics)
    
    # Show reallocated budget
    new_allocation = optimizer.get_current_allocation()
    print(f"\n📊 Reallocated Budget (Google up, Facebook down):")
    for channel, budget in new_allocation.items():
        old_budget = initial_allocation[channel]
        change = budget - old_budget
        change_pct = (change / old_budget * 100) if old_budget > 0 else 0
        print(f"   {channel.value}: ${budget} ({change:+.0f}, {change_pct:+.1f}%)")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"💡 Key Features Demonstrated:")
    print(f"   • Dynamic allocation based on marginal ROAS")
    print(f"   • Crisis time 1.4x multipliers (2am)")
    print(f"   • Decision time 1.5x multipliers (7-9pm)")
    print(f"   • iOS premium bidding (20-30%)")
    print(f"   • Real-time performance-based reallocation")
    print(f"   • Budget pacing to prevent early exhaustion")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demo
    asyncio.run(demo_gaelp_optimizer())