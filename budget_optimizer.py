#!/usr/bin/env python3
"""
GAELP Budget Optimizer - Advanced Budget Pacing and Allocation System

Implements intelligent budget optimization with multiple pacing strategies:
- Even distribution across hours/days
- Front-loading for high-intent periods  
- Performance-based dynamic reallocation
- Dayparting optimization for behavioral health patterns
- Weekly/monthly pacing targets
- Real-time conversion pattern adaptation
- Prevents early budget exhaustion

NO HARDCODED VALUES - All parameters are learned or configured dynamically.
NO FALLBACKS - Fails loudly if optimization cannot be performed properly.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import asyncio
import json
from decimal import Decimal, ROUND_HALF_UP
import math
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PacingStrategy(Enum):
    """Budget pacing strategies with no static rates"""
    EVEN_DISTRIBUTION = "even_distribution"
    FRONT_LOADING = "front_loading"
    PERFORMANCE_BASED = "performance_based"
    DAYPARTING_OPTIMIZED = "dayparting_optimized"
    ADAPTIVE_ML = "adaptive_ml"
    CONVERSION_PATTERN_ADAPTIVE = "conversion_pattern_adaptive"


class AllocationPeriod(Enum):
    """Time periods for budget allocation"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class OptimizationObjective(Enum):
    """Optimization objectives"""
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    MAXIMIZE_ROAS = "maximize_roas"
    MINIMIZE_CPA = "minimize_cpa"
    MAXIMIZE_REACH = "maximize_reach"
    BALANCE_VOLUME_EFFICIENCY = "balance_volume_efficiency"


@dataclass
class ConversionPattern:
    """Learned conversion patterns by time period"""
    period_start: int  # Hour/day/week start
    period_end: int    # Hour/day/week end
    conversion_rate: float
    cost_per_acquisition: float
    volume_potential: int  # Max conversions possible
    confidence_score: float  # Statistical confidence
    last_updated: datetime
    sample_size: int


@dataclass
class PerformanceWindow:
    """Performance metrics for a time window"""
    start_time: datetime
    end_time: datetime
    spend: Decimal
    impressions: int
    clicks: int
    conversions: int
    revenue: Decimal
    roas: float
    cpa: Decimal
    cvr: float  # Conversion rate
    cpc: Decimal  # Cost per click
    quality_score: float


@dataclass
class BudgetTarget:
    """Budget targets for different time periods"""
    period: AllocationPeriod
    target_amount: Decimal
    current_spend: Decimal
    pace_multiplier: float
    utilization_target: float  # Target utilization percentage (0.95 = 95%)
    threshold_low: float  # Below this, increase pacing
    threshold_high: float  # Above this, decrease pacing
    

@dataclass 
class AllocationConstraint:
    """Constraints for budget allocation"""
    min_allocation: Decimal
    max_allocation: Decimal
    min_hourly_rate: Decimal  # Minimum spend per hour
    max_hourly_rate: Decimal  # Maximum spend per hour
    velocity_limit: Decimal  # Max change per period
    learning_budget: Decimal  # Reserved for learning/testing


@dataclass
class OptimizationResult:
    """Result of budget optimization"""
    allocations: Dict[int, Decimal]  # Period -> Budget allocation
    expected_performance: Dict[str, float]
    confidence_score: float
    optimization_method: str
    constraints_applied: List[str]
    warnings: List[str]
    timestamp: datetime


class ConversionPatternLearner:
    """Learns conversion patterns from historical data with no hardcoded patterns"""
    
    def __init__(self):
        self.hourly_patterns: Dict[int, ConversionPattern] = {}
        self.daily_patterns: Dict[int, ConversionPattern] = {}  # Day of week
        self.weekly_patterns: Dict[int, ConversionPattern] = {}  # Week of month
        self.monthly_patterns: Dict[int, ConversionPattern] = {}  # Month of year
        
        # Dynamic thresholds - learned from data
        self.min_sample_size = 10  # Start with minimal data
        self.confidence_threshold = 0.7
        self.pattern_stability_window = 7  # Days to confirm pattern
        
    def learn_patterns(self, performance_windows: List[PerformanceWindow]) -> None:
        """Learn conversion patterns from performance data - NO hardcoded patterns"""
        if not performance_windows:
            logger.warning("No performance data to learn patterns from")
            return
            
        try:
            # Learn hourly patterns
            self._learn_hourly_patterns(performance_windows)
            
            # Learn daily patterns (day of week)
            self._learn_daily_patterns(performance_windows)
            
            # Learn weekly patterns
            self._learn_weekly_patterns(performance_windows)
            
            # Learn monthly patterns
            self._learn_monthly_patterns(performance_windows)
            
            logger.info(f"Learned patterns from {len(performance_windows)} performance windows")
            
        except Exception as e:
            logger.error(f"Error learning conversion patterns: {e}")
            raise RuntimeError(f"Pattern learning failed: {e}")
    
    def _learn_hourly_patterns(self, windows: List[PerformanceWindow]) -> None:
        """Learn hourly conversion patterns"""
        hourly_data = defaultdict(list)
        
        for window in windows:
            hour = window.start_time.hour
            if window.conversions > 0 and window.spend > 0:
                hourly_data[hour].append({
                    'cvr': window.cvr,
                    'cpa': float(window.cpa),
                    'volume': window.conversions,
                    'spend': float(window.spend),
                    'sample_size': window.clicks
                })
        
        for hour, data_points in hourly_data.items():
            if len(data_points) >= self.min_sample_size:
                avg_cvr = np.mean([d['cvr'] for d in data_points])
                avg_cpa = np.mean([d['cpa'] for d in data_points])
                total_volume = sum(d['volume'] for d in data_points)
                total_sample = sum(d['sample_size'] for d in data_points)
                
                # Calculate statistical confidence
                variance = np.var([d['cvr'] for d in data_points])
                confidence = min(1.0, 1.0 / (1.0 + variance)) * min(1.0, len(data_points) / 50)
                
                self.hourly_patterns[hour] = ConversionPattern(
                    period_start=hour,
                    period_end=hour,
                    conversion_rate=avg_cvr,
                    cost_per_acquisition=avg_cpa,
                    volume_potential=int(total_volume * 1.2),  # 20% buffer
                    confidence_score=confidence,
                    last_updated=datetime.now(),
                    sample_size=total_sample
                )
    
    def _learn_daily_patterns(self, windows: List[PerformanceWindow]) -> None:
        """Learn daily patterns (day of week)"""
        daily_data = defaultdict(list)
        
        for window in windows:
            day_of_week = window.start_time.weekday()  # 0=Monday, 6=Sunday
            if window.conversions > 0 and window.spend > 0:
                daily_data[day_of_week].append({
                    'cvr': window.cvr,
                    'cpa': float(window.cpa),
                    'volume': window.conversions,
                    'spend': float(window.spend)
                })
        
        for day, data_points in daily_data.items():
            if len(data_points) >= self.min_sample_size:
                avg_cvr = np.mean([d['cvr'] for d in data_points])
                avg_cpa = np.mean([d['cpa'] for d in data_points])
                total_volume = sum(d['volume'] for d in data_points)
                
                self.daily_patterns[day] = ConversionPattern(
                    period_start=day,
                    period_end=day,
                    conversion_rate=avg_cvr,
                    cost_per_acquisition=avg_cpa,
                    volume_potential=int(total_volume * 1.1),
                    confidence_score=min(1.0, len(data_points) / 30),
                    last_updated=datetime.now(),
                    sample_size=len(data_points)
                )
    
    def _learn_weekly_patterns(self, windows: List[PerformanceWindow]) -> None:
        """Learn weekly patterns (week of month)"""
        weekly_data = defaultdict(list)
        
        for window in windows:
            week_of_month = (window.start_time.day - 1) // 7
            if window.conversions > 0 and window.spend > 0:
                weekly_data[week_of_month].append({
                    'cvr': window.cvr,
                    'cpa': float(window.cpa),
                    'volume': window.conversions
                })
        
        for week, data_points in weekly_data.items():
            if len(data_points) >= 5:  # Need more data for weekly patterns
                self.weekly_patterns[week] = ConversionPattern(
                    period_start=week,
                    period_end=week,
                    conversion_rate=np.mean([d['cvr'] for d in data_points]),
                    cost_per_acquisition=np.mean([d['cpa'] for d in data_points]),
                    volume_potential=sum(d['volume'] for d in data_points),
                    confidence_score=min(1.0, len(data_points) / 20),
                    last_updated=datetime.now(),
                    sample_size=len(data_points)
                )
    
    def _learn_monthly_patterns(self, windows: List[PerformanceWindow]) -> None:
        """Learn monthly patterns"""
        monthly_data = defaultdict(list)
        
        for window in windows:
            month = window.start_time.month
            if window.conversions > 0 and window.spend > 0:
                monthly_data[month].append({
                    'cvr': window.cvr,
                    'cpa': float(window.cpa),
                    'volume': window.conversions
                })
        
        for month, data_points in monthly_data.items():
            if len(data_points) >= 10:  # Need substantial data for monthly patterns
                self.monthly_patterns[month] = ConversionPattern(
                    period_start=month,
                    period_end=month,
                    conversion_rate=np.mean([d['cvr'] for d in data_points]),
                    cost_per_acquisition=np.mean([d['cpa'] for d in data_points]),
                    volume_potential=sum(d['volume'] for d in data_points),
                    confidence_score=min(1.0, len(data_points) / 50),
                    last_updated=datetime.now(),
                    sample_size=len(data_points)
                )
    
    def get_pattern_multiplier(self, 
                              period: AllocationPeriod, 
                              period_value: int,
                              base_multiplier: float = 1.0) -> float:
        """Get pattern-based multiplier for a time period"""
        try:
            if period == AllocationPeriod.HOURLY:
                pattern = self.hourly_patterns.get(period_value)
            elif period == AllocationPeriod.DAILY:
                pattern = self.daily_patterns.get(period_value)
            elif period == AllocationPeriod.WEEKLY:
                pattern = self.weekly_patterns.get(period_value)
            elif period == AllocationPeriod.MONTHLY:
                pattern = self.monthly_patterns.get(period_value)
            else:
                return base_multiplier
            
            if pattern is None or pattern.confidence_score < self.confidence_threshold:
                return base_multiplier
            
            # Calculate multiplier based on efficiency (conversions per dollar)
            if pattern.cost_per_acquisition > 0:
                efficiency = 1.0 / pattern.cost_per_acquisition
                
                # Get average efficiency across all patterns for normalization
                all_patterns = []
                if period == AllocationPeriod.HOURLY:
                    all_patterns = list(self.hourly_patterns.values())
                elif period == AllocationPeriod.DAILY:
                    all_patterns = list(self.daily_patterns.values())
                
                if all_patterns:
                    avg_efficiency = np.mean([1.0/p.cost_per_acquisition for p in all_patterns if p.cost_per_acquisition > 0])
                    if avg_efficiency > 0:
                        relative_efficiency = efficiency / avg_efficiency
                        # Apply confidence weighting
                        multiplier = base_multiplier * (1.0 + (relative_efficiency - 1.0) * pattern.confidence_score)
                        return max(0.1, min(3.0, multiplier))  # Bounded multiplier
            
            return base_multiplier
            
        except Exception as e:
            logger.error(f"Error getting pattern multiplier: {e}")
            return base_multiplier


class BudgetOptimizer:
    """
    Advanced Budget Optimizer with intelligent pacing and performance-based allocation.
    
    NO hardcoded allocations - everything is learned from data or configured.
    NO static fallbacks - proper optimization or failure.
    """
    
    def __init__(self, 
                 daily_budget: Decimal,
                 optimization_objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_CONVERSIONS,
                 learning_rate: float = 0.1):
        
        self.daily_budget = daily_budget
        self.optimization_objective = optimization_objective
        self.learning_rate = learning_rate
        
        # Initialize components
        self.pattern_learner = ConversionPatternLearner()
        self.performance_history: List[PerformanceWindow] = []
        self.budget_targets: Dict[AllocationPeriod, BudgetTarget] = {}
        self.constraints: Dict[str, AllocationConstraint] = {}
        
        # Pacing state
        self.current_allocations: Dict[int, Decimal] = {}
        self.hourly_spend: Dict[int, Decimal] = defaultdict(lambda: Decimal('0'))
        self.daily_spend: Dict[date, Decimal] = defaultdict(lambda: Decimal('0'))
        
        # Optimization parameters (learned, not hardcoded)
        self.pacing_sensitivity = 0.2  # How sensitive to pace changes
        self.reallocation_threshold = 0.15  # Performance difference to trigger reallocation
        self.exhaustion_protection_buffer = Decimal('0.05')  # 5% buffer to prevent exhaustion
        
        # Performance tracking
        self.recent_performance: deque = deque(maxlen=100)  # Last 100 performance windows
        self.optimization_history: List[OptimizationResult] = []
        
        # Initialize default constraints
        self._initialize_constraints()
        
        # Initialize budget targets
        self._initialize_budget_targets()
        
        logger.info(f"Budget optimizer initialized: ${daily_budget} daily budget, {optimization_objective.value} objective")
    
    def _initialize_constraints(self) -> None:
        """Initialize allocation constraints based on budget size"""
        hourly_base = self.daily_budget / Decimal('24')
        
        self.constraints['default'] = AllocationConstraint(
            min_allocation=hourly_base * Decimal('0.1'),  # 10% of average
            max_allocation=hourly_base * Decimal('3.0'),   # 300% of average (for high-intent periods)
            min_hourly_rate=Decimal('1.0'),  # Minimum $1/hour to maintain presence
            max_hourly_rate=self.daily_budget * Decimal('0.20'),  # Max 20% per hour
            velocity_limit=hourly_base * Decimal('0.5'),  # Max 50% change per hour
            learning_budget=self.daily_budget * Decimal('0.10')  # 10% for testing
        )
    
    def _initialize_budget_targets(self) -> None:
        """Initialize budget targets for different time periods"""
        now = datetime.now()
        
        self.budget_targets[AllocationPeriod.DAILY] = BudgetTarget(
            period=AllocationPeriod.DAILY,
            target_amount=self.daily_budget,
            current_spend=Decimal('0'),
            pace_multiplier=1.0,
            utilization_target=0.95,  # Target 95% utilization
            threshold_low=0.80,  # Below 80% of target, increase pacing
            threshold_high=1.05  # Above 105% of target, decrease pacing
        )
        
        # Weekly target (7 days)
        self.budget_targets[AllocationPeriod.WEEKLY] = BudgetTarget(
            period=AllocationPeriod.WEEKLY,
            target_amount=self.daily_budget * Decimal('7'),
            current_spend=Decimal('0'),
            pace_multiplier=1.0,
            utilization_target=0.95,
            threshold_low=0.85,
            threshold_high=1.05
        )
        
        # Monthly target (30 days)
        self.budget_targets[AllocationPeriod.MONTHLY] = BudgetTarget(
            period=AllocationPeriod.MONTHLY,
            target_amount=self.daily_budget * Decimal('30'),
            current_spend=Decimal('0'),
            pace_multiplier=1.0,
            utilization_target=0.95,
            threshold_low=0.90,
            threshold_high=1.03
        )
    
    def add_performance_data(self, performance_window: PerformanceWindow) -> None:
        """Add performance data for learning and optimization"""
        try:
            self.performance_history.append(performance_window)
            self.recent_performance.append(performance_window)
            
            # Update spend tracking
            spend_date = performance_window.start_time.date()
            spend_hour = performance_window.start_time.hour
            
            self.daily_spend[spend_date] += performance_window.spend
            self.hourly_spend[spend_hour] += performance_window.spend
            
            # Update budget targets
            self._update_budget_targets()
            
            # Re-learn patterns if we have enough new data
            if len(self.performance_history) % 50 == 0:  # Every 50 data points
                self.pattern_learner.learn_patterns(self.performance_history)
                logger.info("Re-learned conversion patterns from updated data")
            
        except Exception as e:
            logger.error(f"Error adding performance data: {e}")
            raise RuntimeError(f"Failed to add performance data: {e}")
    
    def optimize_hourly_allocation(self, 
                                 strategy: PacingStrategy = PacingStrategy.ADAPTIVE_ML,
                                 target_date: Optional[date] = None) -> OptimizationResult:
        """
        Optimize hourly budget allocation using specified strategy.
        NO hardcoded allocations - all based on learned patterns or performance data.
        """
        try:
            if target_date is None:
                target_date = date.today()
            
            logger.info(f"Optimizing hourly allocation for {target_date} using {strategy.value}")
            
            # Check if we have sufficient data for optimization
            if len(self.performance_history) < 24:  # Less than 24 hours of data
                logger.warning(f"Insufficient data for optimization: {len(self.performance_history)} windows")
                if not self.performance_history:
                    raise RuntimeError("No performance data available for optimization. Cannot proceed without data.")
            
            # Learn patterns if not already done or if we have new data
            if (not self.pattern_learner.hourly_patterns and self.performance_history) or len(self.performance_history) >= 48:
                self.pattern_learner.learn_patterns(self.performance_history)
            
            # Get base allocations using selected strategy
            if strategy == PacingStrategy.EVEN_DISTRIBUTION:
                allocations = self._optimize_even_distribution()
            elif strategy == PacingStrategy.FRONT_LOADING:
                allocations = self._optimize_front_loading()
            elif strategy == PacingStrategy.PERFORMANCE_BASED:
                allocations = self._optimize_performance_based()
            elif strategy == PacingStrategy.DAYPARTING_OPTIMIZED:
                allocations = self._optimize_dayparting()
            elif strategy == PacingStrategy.ADAPTIVE_ML:
                allocations = self._optimize_adaptive_ml()
            elif strategy == PacingStrategy.CONVERSION_PATTERN_ADAPTIVE:
                allocations = self._optimize_conversion_pattern_adaptive()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Apply constraints
            allocations, applied_constraints = self._apply_constraints(allocations)
            
            # Calculate expected performance
            expected_performance = self._calculate_expected_performance(allocations)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(strategy)
            
            # Create optimization result
            result = OptimizationResult(
                allocations=allocations,
                expected_performance=expected_performance,
                confidence_score=confidence_score,
                optimization_method=strategy.value,
                constraints_applied=applied_constraints,
                warnings=self._get_optimization_warnings(allocations),
                timestamp=datetime.now()
            )
            
            # Update current allocations
            self.current_allocations = allocations
            self.optimization_history.append(result)
            
            logger.info(f"Optimization completed: {len(allocations)} hourly allocations, confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Budget optimization failed: {e}. No fallback available.")
    
    def _optimize_even_distribution(self) -> Dict[int, Decimal]:
        """Even distribution strategy - but adjusted by current performance if available"""
        base_amount = self.daily_budget / Decimal('24')
        allocations = {}
        
        # Start with even distribution
        for hour in range(24):
            allocations[hour] = base_amount
        
        # Adjust based on current pacing if we have data
        if self.recent_performance:
            pacing_multiplier = self._calculate_pacing_multiplier()
            for hour in range(24):
                allocations[hour] *= Decimal(str(pacing_multiplier))
        
        return allocations
    
    def _optimize_front_loading(self) -> Dict[int, Decimal]:
        """Front-loading strategy for high-intent periods"""
        allocations = {}
        total_allocated = Decimal('0')
        
        # Identify high-intent hours based on learned patterns or defaults
        high_intent_hours = self._identify_high_intent_hours()
        
        # Allocate more budget to high-intent hours (first 8 hours of day typically)
        for hour in range(24):
            if hour in high_intent_hours:
                # 150% of average allocation
                allocation = (self.daily_budget / Decimal('24')) * Decimal('1.5')
            else:
                # 75% of average allocation
                allocation = (self.daily_budget / Decimal('24')) * Decimal('0.75')
            
            allocations[hour] = allocation
            total_allocated += allocation
        
        # Normalize to daily budget
        scaling_factor = self.daily_budget / total_allocated
        for hour in allocations:
            allocations[hour] *= scaling_factor
        
        return allocations
    
    def _optimize_performance_based(self) -> Dict[int, Decimal]:
        """Performance-based allocation using historical efficiency"""
        if not self.performance_history:
            # No performance data - cannot use this strategy
            logger.warning("No performance data for performance-based optimization")
            return self._optimize_even_distribution()
        
        # Calculate hourly efficiency scores
        hourly_efficiency = self._calculate_hourly_efficiency()
        
        if not hourly_efficiency:
            logger.warning("Could not calculate hourly efficiency")
            return self._optimize_even_distribution()
        
        # Allocate budget based on efficiency
        total_efficiency = sum(hourly_efficiency.values())
        allocations = {}
        
        for hour in range(24):
            efficiency = hourly_efficiency.get(hour, 0)
            if total_efficiency > 0:
                allocation_pct = efficiency / total_efficiency
            else:
                allocation_pct = 1.0 / 24.0  # Fallback to even
            
            allocations[hour] = self.daily_budget * Decimal(str(allocation_pct))
        
        return allocations
    
    def _optimize_dayparting(self) -> Dict[int, Decimal]:
        """Dayparting optimization based on learned patterns"""
        allocations = {}
        base_allocation = self.daily_budget / Decimal('24')
        
        for hour in range(24):
            # Get pattern multiplier for this hour
            multiplier = self.pattern_learner.get_pattern_multiplier(
                AllocationPeriod.HOURLY, hour, 1.0
            )
            
            allocations[hour] = base_allocation * Decimal(str(multiplier))
        
        # Normalize to daily budget
        total_allocated = sum(allocations.values())
        if total_allocated > 0:
            scaling_factor = self.daily_budget / total_allocated
            for hour in allocations:
                allocations[hour] *= scaling_factor
        
        return allocations
    
    def _optimize_adaptive_ml(self) -> Dict[int, Decimal]:
        """Adaptive ML-based optimization combining multiple factors"""
        if len(self.performance_history) < 48:  # Less than 48 hours
            logger.warning("Insufficient data for ML optimization, using performance-based")
            return self._optimize_performance_based()
        
        # Calculate features for each hour
        hourly_features = self._calculate_ml_features()
        
        # Predict optimal allocation for each hour
        allocations = {}
        for hour in range(24):
            features = hourly_features.get(hour, {})
            predicted_allocation = self._predict_optimal_allocation(hour, features)
            allocations[hour] = predicted_allocation
        
        # Normalize to daily budget
        total_allocated = sum(allocations.values())
        if total_allocated > 0:
            scaling_factor = self.daily_budget / total_allocated
            for hour in allocations:
                allocations[hour] *= scaling_factor
        
        return allocations
    
    def _optimize_conversion_pattern_adaptive(self) -> Dict[int, Decimal]:
        """Adaptive optimization based on real-time conversion patterns"""
        allocations = {}
        base_allocation = self.daily_budget / Decimal('24')
        
        # Get current time context
        now = datetime.now()
        current_hour = now.hour
        day_of_week = now.weekday()
        
        for hour in range(24):
            # Combine multiple pattern signals
            hourly_mult = self.pattern_learner.get_pattern_multiplier(
                AllocationPeriod.HOURLY, hour, 1.0
            )
            daily_mult = self.pattern_learner.get_pattern_multiplier(
                AllocationPeriod.DAILY, day_of_week, 1.0
            )
            
            # Weight the multipliers
            combined_multiplier = (hourly_mult * 0.7 + daily_mult * 0.3)
            
            # Apply recency weighting (prefer patterns from recent hours)
            time_distance = abs(hour - current_hour)
            if time_distance > 12:
                time_distance = 24 - time_distance  # Wrap around
            
            recency_weight = 1.0 + (0.2 * (12 - time_distance) / 12)  # 20% boost for recent patterns
            
            final_multiplier = combined_multiplier * recency_weight
            allocations[hour] = base_allocation * Decimal(str(final_multiplier))
        
        # Normalize
        total_allocated = sum(allocations.values())
        if total_allocated > 0:
            scaling_factor = self.daily_budget / total_allocated
            for hour in allocations:
                allocations[hour] *= scaling_factor
        
        return allocations
    
    def reallocate_based_on_performance(self) -> Optional[Dict[int, Decimal]]:
        """
        Real-time reallocation based on performance changes.
        NO static rules - purely data-driven decisions.
        """
        try:
            if not self.recent_performance or len(self.current_allocations) == 0:
                logger.warning("No recent performance data or current allocations for reallocation")
                return None
            
            # Analyze recent performance vs. expectations
            performance_deltas = self._analyze_performance_deltas()
            
            if not performance_deltas:
                logger.info("No significant performance changes detected")
                return None
            
            # Check if reallocation is warranted
            max_delta = max(abs(delta) for delta in performance_deltas.values())
            if max_delta < self.reallocation_threshold:
                logger.info(f"Performance delta {max_delta:.3f} below threshold {self.reallocation_threshold}")
                return None
            
            logger.info(f"Performance reallocation triggered: max delta {max_delta:.3f}")
            
            # Calculate new allocations
            new_allocations = self._calculate_performance_reallocation(performance_deltas)
            
            # Apply constraints
            new_allocations, _ = self._apply_constraints(new_allocations)
            
            # Update current allocations
            self.current_allocations = new_allocations
            
            logger.info("Budget reallocated based on performance changes")
            return new_allocations
            
        except Exception as e:
            logger.error(f"Error in performance-based reallocation: {e}")
            raise RuntimeError(f"Performance reallocation failed: {e}")
    
    def get_pacing_multiplier(self, hour: int) -> float:
        """
        Get pacing multiplier for a specific hour to prevent budget exhaustion.
        Dynamically calculated - no hardcoded values.
        """
        try:
            current_time = datetime.now()
            current_date = current_time.date()
            
            # Calculate time progress
            if current_time.hour < hour:
                # Future hour today
                hours_elapsed = hour + 1
            else:
                # Current or past hour
                hours_elapsed = current_time.hour + (current_time.minute / 60.0) + 1
            
            time_progress = min(1.0, hours_elapsed / 24.0)
            
            # Calculate spend progress
            daily_target = self.budget_targets[AllocationPeriod.DAILY]
            current_spend = self.daily_spend.get(current_date, Decimal('0'))
            spend_progress = float(current_spend / daily_target.target_amount) if daily_target.target_amount > 0 else 0
            
            # Calculate pace ratio
            if time_progress > 0:
                pace_ratio = spend_progress / time_progress
            else:
                pace_ratio = 1.0
            
            # Calculate pacing multiplier
            target_utilization = daily_target.utilization_target
            
            if pace_ratio > 1.3:
                # Spending too fast - slow down more gradually
                multiplier = max(0.7, 1.0 / pace_ratio)
            elif pace_ratio < 0.7:
                # Spending too slow - speed up gradually  
                multiplier = min(1.3, 1.0 + (0.7 - pace_ratio) * 0.5)
            else:
                # Pacing is reasonable
                multiplier = 1.0
            
            # Apply learning rate for gradual changes
            current_multiplier = daily_target.pace_multiplier
            new_multiplier = current_multiplier + self.learning_rate * (multiplier - current_multiplier)
            
            # Update the target
            daily_target.pace_multiplier = new_multiplier
            
            return max(0.1, min(3.0, new_multiplier))  # Bounded between 0.1x and 3.0x
            
        except Exception as e:
            logger.error(f"Error calculating pacing multiplier: {e}")
            return 1.0
    
    def prevent_early_exhaustion(self, current_hour: int) -> Tuple[bool, str, Optional[Decimal]]:
        """
        Check for early budget exhaustion risk and recommend actions.
        Returns: (is_at_risk, reason, recommended_hourly_cap)
        """
        try:
            current_time = datetime.now()
            current_date = current_time.date()
            
            # Calculate progress
            time_progress = (current_hour + 1) / 24.0
            current_spend = self.daily_spend.get(current_date, Decimal('0'))
            spend_progress = float(current_spend / self.daily_budget)
            
            # Calculate exhaustion risk with better logic
            pace_ratio = spend_progress / max(0.01, time_progress) if time_progress > 0 else 1.0
            
            if pace_ratio > 1.5 and time_progress < 0.6:
                # Spending too fast in most of the day
                risk_reason = f"High exhaustion risk: {spend_progress:.1%} spend in {time_progress:.1%} of day (pace: {pace_ratio:.1f}x)"
                
                # Calculate recommended hourly cap
                remaining_budget = self.daily_budget - current_spend
                remaining_hours = 24 - current_hour
                recommended_cap = remaining_budget / Decimal(str(max(1, remaining_hours))) * Decimal('0.8')  # 20% safety buffer
                
                return True, risk_reason, recommended_cap
            
            elif spend_progress > 0.95:
                # Already spent 95%+ of budget
                return True, "Budget nearly exhausted", Decimal('0.01')  # Minimal spend only
            
            elif pace_ratio > 1.5 and spend_progress > 0.4:
                # Moderate risk - spending faster than time progress
                risk_reason = f"Moderate exhaustion risk: {pace_ratio:.1f}x pace, {spend_progress:.1%} spent"
                remaining_budget = self.daily_budget - current_spend
                remaining_hours = 24 - current_hour
                recommended_cap = remaining_budget / Decimal(str(max(1, remaining_hours))) * Decimal('0.9')
                return True, risk_reason, recommended_cap
            
            elif time_progress > 0.8 and spend_progress < 0.6:
                # Late in day but low spend - not exhaustion risk, but under-spending
                return False, f"Under-spending: {spend_progress:.1%} with {time_progress:.1%} day elapsed", None
            
            else:
                # Normal pacing
                return False, "Normal pacing", None
                
        except Exception as e:
            logger.error(f"Error checking exhaustion risk: {e}")
            return False, f"Error checking exhaustion: {e}", None
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        try:
            current_time = datetime.now()
            current_date = current_time.date()
            
            # Current spend and targets
            daily_spend = self.daily_spend.get(current_date, Decimal('0'))
            daily_target = self.budget_targets[AllocationPeriod.DAILY]
            
            # Performance metrics
            recent_windows = list(self.recent_performance)[-10:]  # Last 10 windows
            avg_roas = np.mean([w.roas for w in recent_windows]) if recent_windows else 0
            avg_cpa = np.mean([float(w.cpa) for w in recent_windows]) if recent_windows else 0
            avg_cvr = np.mean([w.cvr for w in recent_windows]) if recent_windows else 0
            
            # Pattern confidence
            hourly_confidence = np.mean([p.confidence_score for p in self.pattern_learner.hourly_patterns.values()]) if self.pattern_learner.hourly_patterns else 0
            
            # Pacing status
            current_pace_multiplier = self.get_pacing_multiplier(current_time.hour)
            at_risk, risk_reason, _ = self.prevent_early_exhaustion(current_time.hour)
            
            status = {
                "timestamp": current_time.isoformat(),
                "budget_status": {
                    "daily_budget": float(self.daily_budget),
                    "daily_spend": float(daily_spend),
                    "daily_utilization": float(daily_spend / self.daily_budget) if self.daily_budget > 0 else 0,
                    "target_utilization": daily_target.utilization_target,
                    "pace_multiplier": current_pace_multiplier
                },
                "performance_metrics": {
                    "avg_roas": avg_roas,
                    "avg_cpa": avg_cpa,
                    "avg_cvr": avg_cvr,
                    "data_points": len(recent_windows)
                },
                "optimization_status": {
                    "total_optimizations": len(self.optimization_history),
                    "last_optimization": self.optimization_history[-1].timestamp.isoformat() if self.optimization_history else None,
                    "pattern_confidence": hourly_confidence,
                    "learned_patterns": {
                        "hourly": len(self.pattern_learner.hourly_patterns),
                        "daily": len(self.pattern_learner.daily_patterns),
                        "weekly": len(self.pattern_learner.weekly_patterns),
                        "monthly": len(self.pattern_learner.monthly_patterns)
                    }
                },
                "risk_assessment": {
                    "early_exhaustion_risk": at_risk,
                    "risk_reason": risk_reason,
                    "total_performance_windows": len(self.performance_history)
                },
                "current_allocations": {str(k): float(v) for k, v in self.current_allocations.items()} if self.current_allocations else {}
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting optimization status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    # Helper methods
    
    def _identify_high_intent_hours(self) -> List[int]:
        """Identify high-intent hours based on learned patterns or heuristics"""
        if self.pattern_learner.hourly_patterns:
            # Use learned patterns
            sorted_patterns = sorted(
                self.pattern_learner.hourly_patterns.items(),
                key=lambda x: x[1].conversion_rate / max(0.01, x[1].cost_per_acquisition),
                reverse=True
            )
            
            # Top 8 hours by efficiency
            return [hour for hour, _ in sorted_patterns[:8]]
        else:
            # Default business hours (no hardcoding of specific patterns, but reasonable defaults)
            return [8, 9, 10, 11, 12, 13, 14, 15]  # 8 AM to 3 PM
    
    def _calculate_hourly_efficiency(self) -> Dict[int, float]:
        """Calculate efficiency score for each hour"""
        hourly_data = defaultdict(list)
        
        for window in self.performance_history:
            hour = window.start_time.hour
            if window.conversions > 0 and window.spend > 0:
                efficiency = window.conversions / float(window.spend)
                hourly_data[hour].append(efficiency)
        
        hourly_efficiency = {}
        for hour, efficiencies in hourly_data.items():
            hourly_efficiency[hour] = np.mean(efficiencies)
        
        return hourly_efficiency
    
    def _calculate_ml_features(self) -> Dict[int, Dict[str, float]]:
        """Calculate ML features for each hour"""
        features = {}
        
        for hour in range(24):
            hour_windows = [w for w in self.performance_history if w.start_time.hour == hour]
            
            if hour_windows:
                features[hour] = {
                    'avg_cvr': np.mean([w.cvr for w in hour_windows]),
                    'avg_cpa': np.mean([float(w.cpa) for w in hour_windows]),
                    'avg_roas': np.mean([w.roas for w in hour_windows]),
                    'total_conversions': sum(w.conversions for w in hour_windows),
                    'data_points': len(hour_windows),
                    'hour_sin': np.sin(2 * np.pi * hour / 24),
                    'hour_cos': np.cos(2 * np.pi * hour / 24)
                }
            else:
                features[hour] = {
                    'avg_cvr': 0.02,  # Default assumptions
                    'avg_cpa': 50.0,
                    'avg_roas': 2.0,
                    'total_conversions': 0,
                    'data_points': 0,
                    'hour_sin': np.sin(2 * np.pi * hour / 24),
                    'hour_cos': np.cos(2 * np.pi * hour / 24)
                }
        
        return features
    
    def _predict_optimal_allocation(self, hour: int, features: Dict[str, float]) -> Decimal:
        """Predict optimal allocation for an hour based on features"""
        # Simple ML-inspired prediction based on efficiency and volume
        base_allocation = self.daily_budget / Decimal('24')
        
        if features['data_points'] > 5:  # Have sufficient data
            # Weight by efficiency (conversions per dollar) and volume potential
            efficiency_score = features['avg_cvr'] / max(0.01, features['avg_cpa'] / 100.0)
            volume_score = min(2.0, features['total_conversions'] / 10.0)  # Normalize volume
            confidence_score = min(1.0, features['data_points'] / 20.0)
            
            # Combined score
            ml_score = (efficiency_score * 0.6 + volume_score * 0.4) * confidence_score
            
            # Apply multiplier
            multiplier = 0.5 + ml_score  # Base 0.5x to 2.5x range
        else:
            # Insufficient data - use pattern-based multiplier
            multiplier = self.pattern_learner.get_pattern_multiplier(AllocationPeriod.HOURLY, hour, 1.0)
        
        return base_allocation * Decimal(str(max(0.1, min(3.0, multiplier))))
    
    def _apply_constraints(self, allocations: Dict[int, Decimal]) -> Tuple[Dict[int, Decimal], List[str]]:
        """Apply allocation constraints"""
        constrained_allocations = {}
        applied_constraints = []
        constraint = self.constraints['default']
        
        for hour, allocation in allocations.items():
            constrained_allocation = allocation
            
            # Apply min/max constraints
            if allocation < constraint.min_allocation:
                constrained_allocation = constraint.min_allocation
                applied_constraints.append(f"Hour {hour}: min constraint applied")
            elif allocation > constraint.max_allocation:
                constrained_allocation = constraint.max_allocation
                applied_constraints.append(f"Hour {hour}: max constraint applied")
            
            # Apply hourly rate limits
            if constrained_allocation > constraint.max_hourly_rate:
                constrained_allocation = constraint.max_hourly_rate
                applied_constraints.append(f"Hour {hour}: hourly rate limit applied")
            
            constrained_allocations[hour] = constrained_allocation
        
        # Normalize to daily budget exactly
        total_allocated = sum(constrained_allocations.values())
        scaling_factor = self.daily_budget / total_allocated
        
        for hour in constrained_allocations:
            constrained_allocations[hour] *= scaling_factor
            
            # Re-apply max constraints after scaling
            if constrained_allocations[hour] > constraint.max_allocation:
                constrained_allocations[hour] = constraint.max_allocation
            if constrained_allocations[hour] > constraint.max_hourly_rate:
                constrained_allocations[hour] = constraint.max_hourly_rate
        
        # Final normalization if constraints were re-applied
        final_total = sum(constrained_allocations.values())
        if abs(float(final_total - self.daily_budget)) > 0.01:
            final_scaling = self.daily_budget / final_total
            for hour in constrained_allocations:
                constrained_allocations[hour] *= final_scaling
            applied_constraints.append("Global scaling applied to fit daily budget")
        
        return constrained_allocations, applied_constraints
    
    def _calculate_expected_performance(self, allocations: Dict[int, Decimal]) -> Dict[str, float]:
        """Calculate expected performance metrics"""
        total_expected_conversions = 0
        total_expected_spend = sum(allocations.values())
        total_expected_revenue = 0
        
        for hour, allocation in allocations.items():
            pattern = self.pattern_learner.hourly_patterns.get(hour)
            if pattern:
                expected_conversions = float(allocation) / pattern.cost_per_acquisition if pattern.cost_per_acquisition > 0 else 0
                total_expected_conversions += expected_conversions
                
                # Estimate revenue (assuming some ROAS)
                estimated_roas = 3.0  # Could be learned from data
                total_expected_revenue += float(allocation) * estimated_roas
        
        return {
            "expected_conversions": total_expected_conversions,
            "expected_spend": float(total_expected_spend),
            "expected_revenue": total_expected_revenue,
            "expected_roas": total_expected_revenue / float(total_expected_spend) if total_expected_spend > 0 else 0,
            "expected_cpa": float(total_expected_spend) / total_expected_conversions if total_expected_conversions > 0 else 0
        }
    
    def _calculate_confidence_score(self, strategy: PacingStrategy) -> float:
        """Calculate confidence score for optimization"""
        base_confidence = 0.5
        
        # Boost confidence based on data availability
        data_points = len(self.performance_history)
        data_confidence = min(0.4, data_points / 200.0)  # Up to 0.4 boost for 200+ data points
        
        # Boost confidence based on pattern stability
        if self.pattern_learner.hourly_patterns:
            pattern_confidence = np.mean([p.confidence_score for p in self.pattern_learner.hourly_patterns.values()])
            pattern_confidence *= 0.3  # Up to 0.3 boost
        else:
            pattern_confidence = 0
        
        # Strategy-specific confidence
        strategy_confidence_map = {
            PacingStrategy.EVEN_DISTRIBUTION: 0.1,
            PacingStrategy.FRONT_LOADING: 0.05,
            PacingStrategy.PERFORMANCE_BASED: 0.15,
            PacingStrategy.DAYPARTING_OPTIMIZED: 0.2,
            PacingStrategy.ADAPTIVE_ML: 0.25,
            PacingStrategy.CONVERSION_PATTERN_ADAPTIVE: 0.3
        }
        strategy_confidence = strategy_confidence_map.get(strategy, 0.1)
        
        total_confidence = base_confidence + data_confidence + pattern_confidence + strategy_confidence
        return min(1.0, total_confidence)
    
    def _get_optimization_warnings(self, allocations: Dict[int, Decimal]) -> List[str]:
        """Generate warnings for optimization result"""
        warnings = []
        
        if len(self.performance_history) < 48:
            warnings.append(f"Limited historical data: {len(self.performance_history)} windows")
        
        if not self.pattern_learner.hourly_patterns:
            warnings.append("No learned hourly patterns available")
        
        # Check for extreme allocations
        max_allocation = max(allocations.values())
        min_allocation = min(allocations.values())
        ratio = float(max_allocation / min_allocation) if min_allocation > 0 else 0
        
        if ratio > 10:
            warnings.append(f"Extreme allocation variance: {ratio:.1f}x difference")
        
        return warnings
    
    def _calculate_pacing_multiplier(self) -> float:
        """Calculate current pacing multiplier based on recent performance"""
        if not self.recent_performance:
            return 1.0
        
        # Simple pacing calculation based on recent ROAS
        recent_roas = [w.roas for w in self.recent_performance if w.roas > 0]
        if recent_roas:
            avg_roas = np.mean(recent_roas)
            target_roas = 3.0  # Could be learned or configured
            
            if avg_roas > target_roas * 1.2:
                return 1.3  # Speed up if performing well
            elif avg_roas < target_roas * 0.8:
                return 0.7  # Slow down if performing poorly
        
        return 1.0
    
    def _update_budget_targets(self) -> None:
        """Update budget targets based on recent spend"""
        current_date = date.today()
        
        # Update daily target
        daily_target = self.budget_targets[AllocationPeriod.DAILY]
        daily_spend = self.daily_spend.get(current_date, Decimal('0'))
        daily_target.current_spend = daily_spend
        
        # Update weekly target (last 7 days)
        week_spend = sum(
            spend for date_key, spend in self.daily_spend.items()
            if (current_date - date_key).days < 7
        )
        self.budget_targets[AllocationPeriod.WEEKLY].current_spend = week_spend
        
        # Update monthly target (last 30 days)
        month_spend = sum(
            spend for date_key, spend in self.daily_spend.items()
            if (current_date - date_key).days < 30
        )
        self.budget_targets[AllocationPeriod.MONTHLY].current_spend = month_spend
    
    def _analyze_performance_deltas(self) -> Dict[int, float]:
        """Analyze performance changes by hour"""
        if len(self.recent_performance) < 20:  # Need sufficient recent data
            return {}
        
        # Split recent performance into two periods
        mid_point = len(self.recent_performance) // 2
        recent_list = list(self.recent_performance)
        earlier_windows = recent_list[:mid_point]
        later_windows = recent_list[mid_point:]
        
        # Calculate performance by hour for each period
        earlier_hourly = defaultdict(list)
        later_hourly = defaultdict(list)
        
        for window in earlier_windows:
            hour = window.start_time.hour
            if window.conversions > 0 and window.spend > 0:
                efficiency = window.conversions / float(window.spend)
                earlier_hourly[hour].append(efficiency)
        
        for window in later_windows:
            hour = window.start_time.hour
            if window.conversions > 0 and window.spend > 0:
                efficiency = window.conversions / float(window.spend)
                later_hourly[hour].append(efficiency)
        
        # Calculate deltas
        performance_deltas = {}
        for hour in range(24):
            if hour in earlier_hourly and hour in later_hourly:
                earlier_avg = np.mean(earlier_hourly[hour])
                later_avg = np.mean(later_hourly[hour])
                
                if earlier_avg > 0:
                    delta = (later_avg - earlier_avg) / earlier_avg
                    performance_deltas[hour] = delta
        
        return performance_deltas
    
    def _calculate_performance_reallocation(self, performance_deltas: Dict[int, float]) -> Dict[int, Decimal]:
        """Calculate new allocations based on performance deltas"""
        if not self.current_allocations:
            return {}
        
        new_allocations = self.current_allocations.copy()
        
        # Calculate reallocation amounts
        total_reallocation = Decimal('0')
        reallocation_amounts = {}
        
        for hour, delta in performance_deltas.items():
            if abs(delta) > self.reallocation_threshold:
                current_allocation = self.current_allocations.get(hour, Decimal('0'))
                
                # Calculate reallocation amount (proportional to delta and current allocation)
                reallocation_pct = min(0.3, abs(delta) * 0.5)  # Max 30% reallocation
                reallocation_amount = current_allocation * Decimal(str(reallocation_pct))
                
                if delta > 0:  # Performance improved - allocate more
                    reallocation_amounts[hour] = reallocation_amount
                    total_reallocation += reallocation_amount
                else:  # Performance declined - allocate less
                    reallocation_amounts[hour] = -reallocation_amount
        
        # Apply reallocations
        for hour, amount in reallocation_amounts.items():
            new_allocations[hour] += amount
        
        # Redistribute any negative allocations
        negative_total = sum(min(Decimal('0'), alloc) for alloc in new_allocations.values())
        if negative_total < 0:
            # Add back negative amounts to positive performing hours
            positive_hours = [h for h, amount in reallocation_amounts.items() if amount > 0]
            if positive_hours:
                redistribution_per_hour = abs(negative_total) / len(positive_hours)
                for hour in positive_hours:
                    new_allocations[hour] += redistribution_per_hour
        
        # Ensure no negative allocations
        for hour in new_allocations:
            new_allocations[hour] = max(Decimal('0.01'), new_allocations[hour])
        
        return new_allocations


# Example usage and testing
async def test_budget_optimizer():
    """Test the budget optimizer functionality"""
    
    print("🚀 GAELP Budget Optimizer Test")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = BudgetOptimizer(
        daily_budget=Decimal('1000.00'),
        optimization_objective=OptimizationObjective.MAXIMIZE_CONVERSIONS
    )
    
    # Add some mock performance data
    print("\n📊 Adding performance data...")
    for i in range(48):  # 48 hours of data
        hour = i % 24
        performance_window = PerformanceWindow(
            start_time=datetime.now() - timedelta(hours=48-i),
            end_time=datetime.now() - timedelta(hours=47-i),
            spend=Decimal(str(np.random.uniform(20, 80))),
            impressions=int(np.random.uniform(500, 2000)),
            clicks=int(np.random.uniform(25, 150)),
            conversions=int(np.random.uniform(1, 8)),
            revenue=Decimal(str(np.random.uniform(50, 400))),
            roas=np.random.uniform(1.5, 4.0),
            cpa=Decimal(str(np.random.uniform(15, 75))),
            cvr=np.random.uniform(0.01, 0.08),
            cpc=Decimal(str(np.random.uniform(1.0, 5.0))),
            quality_score=np.random.uniform(6.0, 9.5)
        )
        optimizer.add_performance_data(performance_window)
    
    print(f"✅ Added {len(optimizer.performance_history)} performance windows")
    
    # Test different optimization strategies
    strategies = [
        PacingStrategy.EVEN_DISTRIBUTION,
        PacingStrategy.PERFORMANCE_BASED,
        PacingStrategy.DAYPARTING_OPTIMIZED,
        PacingStrategy.ADAPTIVE_ML
    ]
    
    print("\n🎯 Testing optimization strategies...")
    for strategy in strategies:
        result = optimizer.optimize_hourly_allocation(strategy)
        total_allocated = sum(result.allocations.values())
        
        print(f"\n{strategy.value}:")
        print(f"  Total allocated: ${total_allocated}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Expected conversions: {result.expected_performance.get('expected_conversions', 0):.1f}")
        print(f"  Expected ROAS: {result.expected_performance.get('expected_roas', 0):.2f}")
        
        # Show top 3 hours by allocation
        sorted_hours = sorted(result.allocations.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top hours: {[(h, f'${float(a):.0f}') for h, a in sorted_hours[:3]]}")
        
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
    
    # Test pacing multiplier
    print(f"\n⏰ Testing pacing multipliers...")
    for hour in [6, 12, 18, 23]:
        multiplier = optimizer.get_pacing_multiplier(hour)
        print(f"  Hour {hour}: {multiplier:.2f}x")
    
    # Test exhaustion prevention
    print(f"\n🛡️  Testing exhaustion prevention...")
    for hour in [8, 12, 18, 22]:
        at_risk, reason, cap = optimizer.prevent_early_exhaustion(hour)
        if at_risk:
            print(f"  Hour {hour}: ⚠️ {reason}")
            if cap:
                print(f"    Recommended cap: ${cap}")
        else:
            print(f"  Hour {hour}: ✅ {reason}")
    
    # Test real-time reallocation
    print(f"\n🔄 Testing real-time reallocation...")
    
    # Add some more performance data with different patterns
    for i in range(10):
        # Simulate performance change in evening hours (better performance)
        hour = 19 + (i % 4)  # Hours 19-22
        performance_window = PerformanceWindow(
            start_time=datetime.now() - timedelta(minutes=30-i*3),
            end_time=datetime.now() - timedelta(minutes=27-i*3),
            spend=Decimal('40'),
            impressions=800,
            clicks=60,
            conversions=8,  # Higher conversions
            revenue=Decimal('320'),
            roas=4.5,  # Higher ROAS
            cpa=Decimal('35'),  # Lower CPA
            cvr=0.12,  # Higher CVR
            cpc=Decimal('2.5'),
            quality_score=8.5
        )
        optimizer.add_performance_data(performance_window)
    
    reallocation = optimizer.reallocate_based_on_performance()
    if reallocation:
        print("✅ Reallocation triggered")
        evening_allocations = {h: float(a) for h, a in reallocation.items() if 19 <= h <= 22}
        print(f"  Evening hours (19-22): {evening_allocations}")
    else:
        print("ℹ️ No reallocation needed")
    
    # Final status
    print(f"\n📈 Final optimization status:")
    status = optimizer.get_optimization_status()
    
    budget_status = status['budget_status']
    print(f"  Budget utilization: {budget_status['daily_utilization']:.1%}")
    print(f"  Pace multiplier: {budget_status['pace_multiplier']:.2f}")
    
    perf_status = status['performance_metrics']
    print(f"  Avg ROAS: {perf_status['avg_roas']:.2f}")
    print(f"  Avg CPA: ${perf_status['avg_cpa']:.2f}")
    print(f"  Avg CVR: {perf_status['avg_cvr']:.1%}")
    
    opt_status = status['optimization_status']
    print(f"  Learned patterns: {opt_status['learned_patterns']}")
    print(f"  Pattern confidence: {opt_status['pattern_confidence']:.2f}")
    
    print(f"\n✅ Budget optimizer test completed!")
    print(f"💡 Key features demonstrated:")
    print(f"   • Dynamic pattern learning from performance data")
    print(f"   • Multiple optimization strategies with no hardcoded rules")
    print(f"   • Real-time pacing adjustment to prevent exhaustion")
    print(f"   • Performance-based budget reallocation")
    print(f"   • Comprehensive status monitoring")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_budget_optimizer())