#!/usr/bin/env python3
"""
Advanced Budget Pacing System for GAELP
Prevents early budget exhaustion with intelligent pacing algorithms.

Features:
- Hourly budget allocation based on historical patterns
- Intraday pacing with anti-frontloading protection
- Dynamic reallocation based on performance metrics
- Channel-specific budget management
- Circuit breakers for overspending prevention
- Predictive pacing using ML models
- Real-time performance monitoring
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import asyncio
import json
import pickle
from decimal import Decimal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PacingStrategy(Enum):
    """Budget pacing strategies"""
    EVEN_DISTRIBUTION = "even_distribution"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    HISTORICAL_PATTERN = "historical_pattern"
    PREDICTIVE_ML = "predictive_ml"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


class ChannelType(Enum):
    """Marketing channel types"""
    GOOGLE_ADS = "google_ads"
    FACEBOOK_ADS = "facebook_ads"
    TIKTOK_ADS = "tiktok_ads"
    DISPLAY = "display"
    NATIVE = "native"
    VIDEO = "video"
    SEARCH = "search"
    SHOPPING = "shopping"


class CircuitBreakerState(Enum):
    """Circuit breaker states for overspending protection"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Spending blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class HourlyAllocation:
    """Hourly budget allocation configuration"""
    hour: int  # 0-23
    base_allocation_pct: float  # Base percentage of daily budget
    performance_multiplier: float  # Historical performance multiplier
    predicted_conversion_rate: float
    predicted_cost_per_click: float
    confidence_score: float  # ML prediction confidence


@dataclass
class ChannelBudget:
    """Channel-specific budget configuration"""
    channel: ChannelType
    daily_budget: Decimal
    hourly_allocations: List[HourlyAllocation]
    performance_metrics: Dict[str, float]
    spend_velocity_limit: Decimal  # Max spend per minute
    circuit_breaker_threshold: Decimal  # Emergency stop threshold
    last_optimization: datetime
    status: str = "active"


@dataclass
class SpendTransaction:
    """Individual spend transaction for pacing analysis"""
    campaign_id: str
    channel: ChannelType
    amount: Decimal
    timestamp: datetime
    clicks: int = 0
    conversions: int = 0
    cost_per_click: float = 0.0
    conversion_rate: float = 0.0
    quality_score: float = 0.0


@dataclass
class PacingAlert:
    """Budget pacing alert"""
    alert_type: str
    campaign_id: str
    channel: Optional[ChannelType]
    current_spend: Decimal
    budget_consumed_pct: float
    time_elapsed_pct: float
    pace_ratio: float  # spend_pct / time_pct
    projected_overspend: Decimal
    recommended_action: str
    severity: str  # low, medium, high, critical


@dataclass
class CircuitBreaker:
    """Circuit breaker for overspending protection"""
    campaign_id: str
    channel: ChannelType
    state: CircuitBreakerState
    failure_count: int
    last_failure_time: Optional[datetime]
    recovery_timeout: timedelta
    failure_threshold: int = 3
    recovery_test_budget: Decimal = Decimal('10.00')


class HistoricalDataAnalyzer:
    """Analyzes historical spending patterns for predictive pacing"""
    
    def __init__(self):
        self.hourly_patterns: Dict[str, Dict[int, float]] = {}
        self.performance_trends: Dict[str, List[float]] = {}
        self.ml_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
    
    def analyze_historical_patterns(self, transactions: List[SpendTransaction]) -> Dict[int, float]:
        """Analyze spending patterns by hour of day"""
        try:
            if not transactions:
                # Default even distribution
                return {hour: 1/24 for hour in range(24)}
            
            df = pd.DataFrame([{
                'hour': t.timestamp.hour,
                'amount': float(t.amount),
                'conversions': t.conversions,
                'clicks': t.clicks
            } for t in transactions])
            
            # Calculate hourly spend distribution
            hourly_spend = df.groupby('hour')['amount'].sum()
            total_spend = hourly_spend.sum()
            
            if total_spend == 0:
                return {hour: 1/24 for hour in range(24)}
            
            # Normalize to percentages
            hourly_patterns = {}
            for hour in range(24):
                spend = hourly_spend.get(hour, 0)
                hourly_patterns[hour] = spend / total_spend
            
            return hourly_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {e}")
            return {hour: 1/24 for hour in range(24)}
    
    def predict_hourly_performance(self, campaign_id: str, historical_data: List[SpendTransaction]) -> List[HourlyAllocation]:
        """Predict optimal hourly allocations using ML"""
        try:
            if len(historical_data) < 168:  # Less than 1 week of hourly data
                logger.warning(f"Insufficient data for ML prediction for {campaign_id}")
                return self._generate_default_allocations()
            
            # Prepare features
            features = []
            targets_ctr = []
            targets_cpc = []
            
            hourly_data = {}
            for transaction in historical_data:
                hour = transaction.timestamp.hour
                if hour not in hourly_data:
                    hourly_data[hour] = []
                hourly_data[hour].append(transaction)
            
            for hour in range(24):
                if hour not in hourly_data:
                    continue
                    
                hour_transactions = hourly_data[hour]
                total_clicks = sum(t.clicks for t in hour_transactions)
                total_conversions = sum(t.conversions for t in hour_transactions)
                total_spend = sum(float(t.amount) for t in hour_transactions)
                
                if total_clicks > 0 and total_spend > 0:
                    features.append([
                        hour,
                        len(hour_transactions),  # Number of transactions
                        np.sin(2 * np.pi * hour / 24),  # Hour cyclical feature
                        np.cos(2 * np.pi * hour / 24),
                    ])
                    targets_ctr.append(total_conversions / total_clicks if total_clicks > 0 else 0)
                    targets_cpc.append(total_spend / total_clicks if total_clicks > 0 else 0)
            
            if len(features) < 10:
                logger.warning(f"Insufficient feature data for ML prediction for {campaign_id}")
                return self._generate_default_allocations()
            
            # Train models
            X = np.array(features)
            y_ctr = np.array(targets_ctr)
            y_cpc = np.array(targets_cpc)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train models
            ctr_model = LinearRegression()
            cpc_model = LinearRegression()
            
            ctr_model.fit(X_scaled, y_ctr)
            cpc_model.fit(X_scaled, y_cpc)
            
            # Store models
            self.ml_models[f"{campaign_id}_ctr"] = ctr_model
            self.ml_models[f"{campaign_id}_cpc"] = cpc_model
            self.scalers[campaign_id] = scaler
            
            # Generate predictions for all 24 hours
            allocations = []
            for hour in range(24):
                features_hour = np.array([[
                    hour,
                    len(hourly_data.get(hour, [])),
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                ]])
                
                features_scaled = scaler.transform(features_hour)
                predicted_ctr = max(0, ctr_model.predict(features_scaled)[0])
                predicted_cpc = max(0.01, cpc_model.predict(features_scaled)[0])
                
                # Calculate performance score
                performance_score = predicted_ctr / predicted_cpc if predicted_cpc > 0 else 0
                base_allocation = 1/24  # Start with even distribution
                
                allocations.append(HourlyAllocation(
                    hour=hour,
                    base_allocation_pct=base_allocation,
                    performance_multiplier=performance_score,
                    predicted_conversion_rate=predicted_ctr,
                    predicted_cost_per_click=predicted_cpc,
                    confidence_score=min(1.0, len(hourly_data.get(hour, [])) / 10)
                ))
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error in ML prediction for {campaign_id}: {e}")
            return self._generate_default_allocations()
    
    def _generate_default_allocations(self) -> List[HourlyAllocation]:
        """Generate default hourly allocations"""
        # Business hours get higher allocation
        business_hours = {9: 1.5, 10: 1.8, 11: 2.0, 12: 1.8, 13: 1.5, 14: 1.8, 
                         15: 2.0, 16: 1.8, 17: 1.5, 18: 1.3, 19: 1.2, 20: 1.1}
        
        total_weight = sum(business_hours.get(h, 0.8) for h in range(24))
        
        allocations = []
        for hour in range(24):
            weight = business_hours.get(hour, 0.8)
            allocation_pct = weight / total_weight
            
            allocations.append(HourlyAllocation(
                hour=hour,
                base_allocation_pct=allocation_pct,
                performance_multiplier=1.0,
                predicted_conversion_rate=0.02,  # Default 2% CTR
                predicted_cost_per_click=1.50,   # Default $1.50 CPC
                confidence_score=0.5
            ))
        
        return allocations


class BudgetPacer:
    """
    Advanced budget pacing system that prevents early exhaustion
    and optimizes spend distribution throughout the day.
    """
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.channel_budgets: Dict[str, Dict[ChannelType, ChannelBudget]] = {}
        self.spend_history: Dict[str, List[SpendTransaction]] = {}
        self.circuit_breakers: Dict[str, Dict[ChannelType, CircuitBreaker]] = {}
        self.pacing_alerts: List[PacingAlert] = []
        self.historical_analyzer = HistoricalDataAnalyzer()
        self.alert_callback = alert_callback
        
        # Pacing parameters
        self.max_hourly_spend_pct = 0.15  # Max 15% of daily budget in one hour
        self.frontload_protection_hours = 4  # First 4 hours have stricter limits
        self.performance_lookback_days = 14  # Days to look back for performance data
        self.reallocation_threshold = 0.20  # 20% performance difference triggers reallocation
        
        # Safety parameters
        self.emergency_stop_threshold = 0.90  # Stop at 90% of daily budget
        self.pace_warning_threshold = 1.5  # Alert when pace > 150% of target
        self.pace_critical_threshold = 2.0  # Critical alert when pace > 200%
        
        logger.info("Budget pacer initialized with advanced pacing algorithms")
    
    def allocate_hourly_budget(self, 
                              campaign_id: str, 
                              channel: ChannelType, 
                              daily_budget: Decimal,
                              strategy: PacingStrategy = PacingStrategy.ADAPTIVE_HYBRID) -> List[HourlyAllocation]:
        """
        Allocate daily budget across 24 hours based on selected strategy
        """
        try:
            # Initialize channel budget if not exists
            if campaign_id not in self.channel_budgets:
                self.channel_budgets[campaign_id] = {}
            
            # Get historical data for this campaign/channel
            historical_data = self._get_historical_data(campaign_id, channel)
            
            # Generate allocations based on strategy
            if strategy == PacingStrategy.EVEN_DISTRIBUTION:
                allocations = self._allocate_evenly()
            elif strategy == PacingStrategy.PERFORMANCE_WEIGHTED:
                allocations = self._allocate_by_performance(historical_data)
            elif strategy == PacingStrategy.HISTORICAL_PATTERN:
                allocations = self._allocate_by_historical_pattern(historical_data)
            elif strategy == PacingStrategy.PREDICTIVE_ML:
                allocations = self.historical_analyzer.predict_hourly_performance(
                    campaign_id, historical_data)
            else:  # ADAPTIVE_HYBRID
                allocations = self._allocate_adaptive_hybrid(campaign_id, historical_data)
            
            # Apply safety constraints
            allocations = self._apply_safety_constraints(allocations)
            
            # Create channel budget
            channel_budget = ChannelBudget(
                channel=channel,
                daily_budget=daily_budget,
                hourly_allocations=allocations,
                performance_metrics=self._calculate_performance_metrics(historical_data),
                spend_velocity_limit=daily_budget / (24 * 60),  # Per minute limit
                circuit_breaker_threshold=daily_budget * Decimal('0.90'),
                last_optimization=datetime.utcnow()
            )
            
            self.channel_budgets[campaign_id][channel] = channel_budget
            
            # Initialize circuit breaker
            if campaign_id not in self.circuit_breakers:
                self.circuit_breakers[campaign_id] = {}
            
            self.circuit_breakers[campaign_id][channel] = CircuitBreaker(
                campaign_id=campaign_id,
                channel=channel,
                state=CircuitBreakerState.CLOSED,
                failure_count=0,
                last_failure_time=None,
                recovery_timeout=timedelta(minutes=30)
            )
            
            logger.info(f"Allocated hourly budget for {campaign_id}/{channel.value}: {strategy.value}")
            return allocations
            
        except Exception as e:
            logger.error(f"Error allocating hourly budget: {e}")
            return self._allocate_evenly()
    
    def check_pace(self, campaign_id: str, channel: ChannelType) -> Tuple[float, Optional[PacingAlert]]:
        """
        Check current spending pace and return pace ratio and alert if needed
        Pace ratio: (spend_percentage / time_percentage)
        - 1.0 = perfect pace
        - > 1.0 = spending too fast
        - < 1.0 = spending too slow
        """
        try:
            if (campaign_id not in self.channel_budgets or 
                channel not in self.channel_budgets[campaign_id]):
                logger.warning(f"No budget found for {campaign_id}/{channel.value}")
                return 1.0, None
            
            channel_budget = self.channel_budgets[campaign_id][channel]
            current_time = datetime.utcnow()
            
            # Calculate time elapsed percentage for today
            start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            time_elapsed = current_time - start_of_day
            time_elapsed_pct = time_elapsed.total_seconds() / (24 * 3600)
            
            # Calculate spend percentage
            today_spend = self._calculate_today_spend(campaign_id, channel)
            spend_pct = float(today_spend / channel_budget.daily_budget) if channel_budget.daily_budget > 0 else 0
            
            # Calculate pace ratio
            pace_ratio = spend_pct / time_elapsed_pct if time_elapsed_pct > 0 else 0
            
            # Check for alerts
            alert = None
            if pace_ratio > self.pace_critical_threshold:
                alert = self._create_pacing_alert(
                    "critical_overpacing", campaign_id, channel, today_spend,
                    spend_pct, time_elapsed_pct, pace_ratio, "critical"
                )
            elif pace_ratio > self.pace_warning_threshold:
                alert = self._create_pacing_alert(
                    "warning_overpacing", campaign_id, channel, today_spend,
                    spend_pct, time_elapsed_pct, pace_ratio, "medium"
                )
            elif spend_pct > self.emergency_stop_threshold:
                alert = self._create_pacing_alert(
                    "emergency_stop_required", campaign_id, channel, today_spend,
                    spend_pct, time_elapsed_pct, pace_ratio, "critical"
                )
            
            if alert:
                asyncio.create_task(self._handle_pacing_alert(alert))
            
            return pace_ratio, alert
            
        except Exception as e:
            logger.error(f"Error checking pace for {campaign_id}/{channel.value}: {e}")
            return 1.0, None
    
    async def reallocate_unused(self, campaign_id: str) -> Dict[ChannelType, Decimal]:
        """
        Reallocate unused budget from underperforming channels to high-performers
        """
        try:
            if campaign_id not in self.channel_budgets:
                logger.warning(f"No budgets found for campaign {campaign_id}")
                return {}
            
            current_time = datetime.utcnow()
            reallocation_results = {}
            
            # Analyze performance for each channel
            channel_performance = {}
            for channel, budget in self.channel_budgets[campaign_id].items():
                performance = self._analyze_channel_performance(campaign_id, channel)
                channel_performance[channel] = performance
            
            # Identify underperformers and overperformers
            avg_performance = np.mean(list(channel_performance.values()))
            underperformers = {ch: perf for ch, perf in channel_performance.items() 
                             if perf < avg_performance * (1 - self.reallocation_threshold)}
            overperformers = {ch: perf for ch, perf in channel_performance.items() 
                            if perf > avg_performance * (1 + self.reallocation_threshold)}
            
            if not underperformers or not overperformers:
                logger.info(f"No significant performance gaps found for {campaign_id}")
                return {}
            
            # Calculate unused budget from underperformers
            total_unused = Decimal('0')
            for channel in underperformers:
                unused = self._calculate_unused_budget(campaign_id, channel)
                total_unused += unused
                reallocation_results[channel] = -unused  # Negative indicates reduction
            
            # Distribute unused budget to overperformers
            if total_unused > 0:
                overperformer_weights = {ch: perf for ch, perf in overperformers.items()}
                total_weight = sum(overperformer_weights.values())
                
                for channel, weight in overperformer_weights.items():
                    additional_budget = total_unused * Decimal(str(weight / total_weight))
                    reallocation_results[channel] = additional_budget
                    
                    # Update channel budget
                    if channel in self.channel_budgets[campaign_id]:
                        self.channel_budgets[campaign_id][channel].daily_budget += additional_budget
            
            logger.info(f"Reallocated {total_unused} across {len(reallocation_results)} channels for {campaign_id}")
            return reallocation_results
            
        except Exception as e:
            logger.error(f"Error reallocating budget for {campaign_id}: {e}")
            return {}
    
    async def emergency_stop(self, campaign_id: str, reason: str = "Emergency stop triggered") -> bool:
        """
        Emergency stop all spending for a campaign
        """
        try:
            if campaign_id not in self.circuit_breakers:
                logger.warning(f"No circuit breakers found for campaign {campaign_id}")
                return False
            
            stopped_channels = 0
            for channel, breaker in self.circuit_breakers[campaign_id].items():
                breaker.state = CircuitBreakerState.OPEN
                breaker.last_failure_time = datetime.utcnow()
                breaker.failure_count += 1
                stopped_channels += 1
                
                logger.critical(f"Emergency stop activated: {campaign_id}/{channel.value} - {reason}")
            
            # Create critical alert
            alert = PacingAlert(
                alert_type="emergency_stop",
                campaign_id=campaign_id,
                channel=None,
                current_spend=self._calculate_total_today_spend(campaign_id),
                budget_consumed_pct=1.0,
                time_elapsed_pct=self._get_time_elapsed_pct(),
                pace_ratio=999.0,  # Indicates emergency
                projected_overspend=Decimal('0'),
                recommended_action="All spending stopped - manual review required",
                severity="critical"
            )
            
            await self._handle_pacing_alert(alert)
            
            logger.critical(f"Emergency stop completed: {stopped_channels} channels stopped for {campaign_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in emergency stop for {campaign_id}: {e}")
            return False
    
    def can_spend(self, campaign_id: str, channel: ChannelType, amount: Decimal) -> Tuple[bool, str]:
        """
        Check if a spend amount is allowed based on pacing rules
        """
        try:
            # Check circuit breaker
            if (campaign_id in self.circuit_breakers and 
                channel in self.circuit_breakers[campaign_id]):
                breaker = self.circuit_breakers[campaign_id][channel]
                if breaker.state == CircuitBreakerState.OPEN:
                    return False, "Circuit breaker is open - spending blocked"
            
            # Check if channel budget exists
            if (campaign_id not in self.channel_budgets or 
                channel not in self.channel_budgets[campaign_id]):
                return False, "No budget allocation found"
            
            channel_budget = self.channel_budgets[campaign_id][channel]
            
            # Check daily budget limit
            today_spend = self._calculate_today_spend(campaign_id, channel)
            if today_spend + amount > channel_budget.daily_budget:
                return False, "Would exceed daily budget limit"
            
            # Check hourly pacing limit
            current_hour = datetime.utcnow().hour
            hourly_allocation = next(
                (a for a in channel_budget.hourly_allocations if a.hour == current_hour),
                None
            )
            
            if hourly_allocation:
                hourly_budget = channel_budget.daily_budget * Decimal(str(hourly_allocation.base_allocation_pct))
                hour_spend = self._calculate_current_hour_spend(campaign_id, channel)
                
                if hour_spend + amount > hourly_budget:
                    return False, "Would exceed hourly pacing limit"
            
            # Check velocity limit
            if amount > channel_budget.spend_velocity_limit:
                return False, "Exceeds spend velocity limit"
            
            # Check frontload protection (first 4 hours of day)
            current_time = datetime.utcnow()
            if current_time.hour < self.frontload_protection_hours:
                max_early_spend = channel_budget.daily_budget * Decimal(str(self.max_hourly_spend_pct))
                if today_spend + amount > max_early_spend * (current_time.hour + 1):
                    return False, "Frontload protection - would spend too much too early"
            
            return True, "Spend approved"
            
        except Exception as e:
            logger.error(f"Error checking spend permission: {e}")
            return False, f"Error in spend validation: {e}"
    
    def record_spend(self, transaction: SpendTransaction):
        """Record a spend transaction for pacing analysis"""
        try:
            if transaction.campaign_id not in self.spend_history:
                self.spend_history[transaction.campaign_id] = []
            
            self.spend_history[transaction.campaign_id].append(transaction)
            
            # Trigger circuit breaker check
            self._check_circuit_breaker(transaction)
            
            logger.debug(f"Recorded spend: {transaction.amount} for {transaction.campaign_id}/{transaction.channel.value}")
            
        except Exception as e:
            logger.error(f"Error recording spend transaction: {e}")
    
    def get_pacing_multiplier(self, hour: int, spent_so_far: float, daily_budget: float) -> float:
        """
        Calculate pacing multiplier based on time of day and spend progress.
        This is a compatibility method for tests.
        
        Args:
            hour: Hour of day (0-23)
            spent_so_far: Amount spent so far today
            daily_budget: Total daily budget
            
        Returns:
            Pacing multiplier (0.0 to 2.0)
        """
        # Calculate progress through day
        hours_elapsed = hour + 1
        hours_remaining = 24 - hour
        day_progress = hours_elapsed / 24.0
        
        # Calculate spend progress
        spend_progress = spent_so_far / daily_budget if daily_budget > 0 else 0
        
        # If we're ahead of schedule, slow down
        if spend_progress > day_progress:
            # Over-spending - reduce multiplier
            multiplier = 0.5 * (1.0 - (spend_progress - day_progress))
        else:
            # Under-spending or on track
            if hours_remaining < 6 and spend_progress < 0.8:
                # Few hours left and budget remaining - speed up
                multiplier = 1.5 + (0.5 * (1.0 - spend_progress))
            elif hour >= 10 and hour <= 22:
                # Peak hours - normal pacing
                multiplier = 1.0
            elif hour < 6:
                # Early morning - slow pacing
                multiplier = 0.7
            else:
                # Default pacing
                multiplier = 1.0
        
        # Ensure multiplier is in valid range
        return max(0.1, min(2.0, multiplier))
    
    # Helper methods
    
    def _get_historical_data(self, campaign_id: str, channel: ChannelType) -> List[SpendTransaction]:
        """Get historical spend data for analysis"""
        if campaign_id not in self.spend_history:
            return []
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.performance_lookback_days)
        return [t for t in self.spend_history[campaign_id] 
                if t.channel == channel and t.timestamp >= cutoff_date]
    
    def _allocate_evenly(self) -> List[HourlyAllocation]:
        """Allocate budget evenly across 24 hours"""
        allocations = []
        for hour in range(24):
            allocations.append(HourlyAllocation(
                hour=hour,
                base_allocation_pct=1/24,
                performance_multiplier=1.0,
                predicted_conversion_rate=0.02,
                predicted_cost_per_click=1.50,
                confidence_score=1.0
            ))
        return allocations
    
    def _allocate_by_performance(self, historical_data: List[SpendTransaction]) -> List[HourlyAllocation]:
        """Allocate based on historical performance"""
        if not historical_data:
            return self._allocate_evenly()
        
        # Calculate performance by hour
        hourly_performance = {}
        for transaction in historical_data:
            hour = transaction.timestamp.hour
            if hour not in hourly_performance:
                hourly_performance[hour] = {'spend': 0, 'conversions': 0, 'clicks': 0}
            
            hourly_performance[hour]['spend'] += float(transaction.amount)
            hourly_performance[hour]['conversions'] += transaction.conversions
            hourly_performance[hour]['clicks'] += transaction.clicks
        
        # Calculate efficiency scores
        efficiency_scores = {}
        for hour, data in hourly_performance.items():
            if data['spend'] > 0:
                cpa = data['spend'] / data['conversions'] if data['conversions'] > 0 else float('inf')
                efficiency_scores[hour] = 1 / cpa if cpa != float('inf') else 0
            else:
                efficiency_scores[hour] = 0
        
        # Normalize to percentages
        total_efficiency = sum(efficiency_scores.values())
        if total_efficiency == 0:
            return self._allocate_evenly()
        
        allocations = []
        for hour in range(24):
            efficiency = efficiency_scores.get(hour, 0)
            allocation_pct = efficiency / total_efficiency if total_efficiency > 0 else 1/24
            
            allocations.append(HourlyAllocation(
                hour=hour,
                base_allocation_pct=allocation_pct,
                performance_multiplier=efficiency,
                predicted_conversion_rate=0.02,  # Will be refined by ML
                predicted_cost_per_click=1.50,
                confidence_score=min(1.0, len([t for t in historical_data if t.timestamp.hour == hour]) / 10)
            ))
        
        return allocations
    
    def _allocate_by_historical_pattern(self, historical_data: List[SpendTransaction]) -> List[HourlyAllocation]:
        """Allocate based on historical spending patterns"""
        patterns = self.historical_analyzer.analyze_historical_patterns(historical_data)
        
        allocations = []
        for hour in range(24):
            allocation_pct = patterns.get(hour, 1/24)
            
            allocations.append(HourlyAllocation(
                hour=hour,
                base_allocation_pct=allocation_pct,
                performance_multiplier=1.0,
                predicted_conversion_rate=0.02,
                predicted_cost_per_click=1.50,
                confidence_score=0.7
            ))
        
        return allocations
    
    def _allocate_adaptive_hybrid(self, campaign_id: str, historical_data: List[SpendTransaction]) -> List[HourlyAllocation]:
        """Adaptive hybrid allocation combining multiple strategies"""
        # Get allocations from different strategies
        even_alloc = self._allocate_evenly()
        perf_alloc = self._allocate_by_performance(historical_data)
        pattern_alloc = self._allocate_by_historical_pattern(historical_data)
        ml_alloc = self.historical_analyzer.predict_hourly_performance(campaign_id, historical_data)
        
        # Weight strategies based on data availability and confidence
        data_points = len(historical_data)
        if data_points < 24:  # Less than 1 day
            weights = [0.8, 0.1, 0.1, 0.0]  # Mostly even
        elif data_points < 168:  # Less than 1 week
            weights = [0.4, 0.3, 0.3, 0.0]  # Mix of even and patterns
        else:  # Sufficient data
            weights = [0.1, 0.3, 0.3, 0.3]  # Favor advanced methods
        
        # Combine allocations
        hybrid_allocations = []
        for hour in range(24):
            combined_allocation = (
                even_alloc[hour].base_allocation_pct * weights[0] +
                perf_alloc[hour].base_allocation_pct * weights[1] +
                pattern_alloc[hour].base_allocation_pct * weights[2] +
                ml_alloc[hour].base_allocation_pct * weights[3]
            )
            
            hybrid_allocations.append(HourlyAllocation(
                hour=hour,
                base_allocation_pct=combined_allocation,
                performance_multiplier=ml_alloc[hour].performance_multiplier,
                predicted_conversion_rate=ml_alloc[hour].predicted_conversion_rate,
                predicted_cost_per_click=ml_alloc[hour].predicted_cost_per_click,
                confidence_score=min(1.0, data_points / 168)  # Confidence increases with data
            ))
        
        return hybrid_allocations
    
    def _apply_safety_constraints(self, allocations: List[HourlyAllocation]) -> List[HourlyAllocation]:
        """Apply safety constraints to prevent frontloading"""
        constrained_allocations = []
        
        for allocation in allocations:
            # Apply frontload protection
            if allocation.hour < self.frontload_protection_hours:
                max_allowed = self.max_hourly_spend_pct * 0.5  # Even stricter in early hours
                allocation.base_allocation_pct = min(allocation.base_allocation_pct, max_allowed)
            else:
                # Regular hourly limit
                allocation.base_allocation_pct = min(allocation.base_allocation_pct, self.max_hourly_spend_pct)
            
            constrained_allocations.append(allocation)
        
        # Renormalize to ensure they sum to 1.0
        total_allocation = sum(a.base_allocation_pct for a in constrained_allocations)
        if total_allocation > 0:
            for allocation in constrained_allocations:
                allocation.base_allocation_pct /= total_allocation
        
        return constrained_allocations
    
    def _calculate_today_spend(self, campaign_id: str, channel: ChannelType) -> Decimal:
        """Calculate total spend for today"""
        if campaign_id not in self.spend_history:
            return Decimal('0')
        
        today = datetime.utcnow().date()
        today_spend = Decimal('0')
        
        for transaction in self.spend_history[campaign_id]:
            if transaction.channel == channel and transaction.timestamp.date() == today:
                today_spend += transaction.amount
        
        return today_spend
    
    def _calculate_current_hour_spend(self, campaign_id: str, channel: ChannelType) -> Decimal:
        """Calculate spend for current hour"""
        if campaign_id not in self.spend_history:
            return Decimal('0')
        
        current_time = datetime.utcnow()
        current_hour_start = current_time.replace(minute=0, second=0, microsecond=0)
        hour_spend = Decimal('0')
        
        for transaction in self.spend_history[campaign_id]:
            if (transaction.channel == channel and 
                transaction.timestamp >= current_hour_start and
                transaction.timestamp < current_hour_start + timedelta(hours=1)):
                hour_spend += transaction.amount
        
        return hour_spend
    
    def _calculate_performance_metrics(self, historical_data: List[SpendTransaction]) -> Dict[str, float]:
        """Calculate channel performance metrics"""
        if not historical_data:
            return {}
        
        total_spend = sum(float(t.amount) for t in historical_data)
        total_clicks = sum(t.clicks for t in historical_data)
        total_conversions = sum(t.conversions for t in historical_data)
        
        return {
            'total_spend': total_spend,
            'total_clicks': total_clicks,
            'total_conversions': total_conversions,
            'ctr': total_conversions / total_clicks if total_clicks > 0 else 0,
            'cpc': total_spend / total_clicks if total_clicks > 0 else 0,
            'cpa': total_spend / total_conversions if total_conversions > 0 else 0,
            'roas': 0  # Would need revenue data
        }
    
    def _create_pacing_alert(self, alert_type: str, campaign_id: str, channel: ChannelType,
                           current_spend: Decimal, spend_pct: float, time_pct: float,
                           pace_ratio: float, severity: str) -> PacingAlert:
        """Create a pacing alert"""
        daily_budget = self.channel_budgets[campaign_id][channel].daily_budget
        projected_overspend = max(Decimal('0'), 
                                current_spend + (current_spend * Decimal(str(1/time_pct - 1))) - daily_budget)
        
        if pace_ratio > 2.0:
            action = "Emergency stop recommended"
        elif pace_ratio > 1.5:
            action = "Reduce bid modifiers by 30%"
        else:
            action = "Monitor closely"
        
        return PacingAlert(
            alert_type=alert_type,
            campaign_id=campaign_id,
            channel=channel,
            current_spend=current_spend,
            budget_consumed_pct=spend_pct,
            time_elapsed_pct=time_pct,
            pace_ratio=pace_ratio,
            projected_overspend=projected_overspend,
            recommended_action=action,
            severity=severity
        )
    
    def _check_circuit_breaker(self, transaction: SpendTransaction):
        """Check and update circuit breaker state"""
        try:
            campaign_id = transaction.campaign_id
            channel = transaction.channel
            
            if (campaign_id not in self.circuit_breakers or 
                channel not in self.circuit_breakers[campaign_id]):
                return
            
            breaker = self.circuit_breakers[campaign_id][channel]
            channel_budget = self.channel_budgets[campaign_id][channel]
            
            # Check if spending is approaching threshold
            today_spend = self._calculate_today_spend(campaign_id, channel)
            if today_spend + transaction.amount > channel_budget.circuit_breaker_threshold:
                breaker.failure_count += 1
                breaker.last_failure_time = datetime.utcnow()
                
                if breaker.failure_count >= breaker.failure_threshold:
                    breaker.state = CircuitBreakerState.OPEN
                    logger.critical(f"Circuit breaker OPENED for {campaign_id}/{channel.value}")
                    
                    # Schedule recovery attempt
                    asyncio.create_task(self._schedule_recovery(breaker))
        
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
    
    async def _schedule_recovery(self, breaker: CircuitBreaker):
        """Schedule circuit breaker recovery attempt"""
        try:
            await asyncio.sleep(breaker.recovery_timeout.total_seconds())
            
            if breaker.state == CircuitBreakerState.OPEN:
                breaker.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker HALF_OPEN for {breaker.campaign_id}/{breaker.channel.value}")
                
                # Test with small budget
                # In production, this would trigger a small test spend
                # For now, we'll just log the attempt
                logger.info(f"Testing recovery with {breaker.recovery_test_budget} budget")
                
                # Simulate successful test (in production, this would be actual performance)
                await asyncio.sleep(60)  # Wait for test results
                
                # If test successful, close breaker
                breaker.state = CircuitBreakerState.CLOSED
                breaker.failure_count = 0
                logger.info(f"Circuit breaker CLOSED for {breaker.campaign_id}/{breaker.channel.value}")
        
        except Exception as e:
            logger.error(f"Error in circuit breaker recovery: {e}")
    
    async def _handle_pacing_alert(self, alert: PacingAlert):
        """Handle pacing alerts"""
        try:
            self.pacing_alerts.append(alert)
            
            if self.alert_callback:
                await self.alert_callback(alert)
            
            logger.warning(f"Pacing alert: {alert.alert_type} for {alert.campaign_id} - {alert.recommended_action}")
        
        except Exception as e:
            logger.error(f"Error handling pacing alert: {e}")
    
    def _analyze_channel_performance(self, campaign_id: str, channel: ChannelType) -> float:
        """Analyze channel performance score"""
        try:
            historical_data = self._get_historical_data(campaign_id, channel)
            if not historical_data:
                return 0.5  # Neutral score
            
            metrics = self._calculate_performance_metrics(historical_data)
            
            # Simple performance score (higher is better)
            # In production, this would be more sophisticated
            ctr = metrics.get('ctr', 0)
            cpa = metrics.get('cpa', float('inf'))
            
            if cpa == float('inf') or cpa == 0:
                return 0.1
            
            # Performance score based on CTR and inverse CPA
            score = ctr * (1 / cpa) * 1000  # Scale factor
            return min(1.0, max(0.0, score))
        
        except Exception as e:
            logger.error(f"Error analyzing channel performance: {e}")
            return 0.5
    
    def _calculate_unused_budget(self, campaign_id: str, channel: ChannelType) -> Decimal:
        """Calculate unused budget for a channel"""
        try:
            if (campaign_id not in self.channel_budgets or 
                channel not in self.channel_budgets[campaign_id]):
                return Decimal('0')
            
            channel_budget = self.channel_budgets[campaign_id][channel]
            today_spend = self._calculate_today_spend(campaign_id, channel)
            current_time = datetime.utcnow()
            
            # Calculate expected spend based on time elapsed
            time_elapsed_pct = self._get_time_elapsed_pct()
            expected_spend = channel_budget.daily_budget * Decimal(str(time_elapsed_pct))
            
            # If we're spending less than 80% of expected, consider it unused
            if today_spend < expected_spend * Decimal('0.8'):
                return expected_spend - today_spend
            
            return Decimal('0')
        
        except Exception as e:
            logger.error(f"Error calculating unused budget: {e}")
            return Decimal('0')
    
    def _calculate_total_today_spend(self, campaign_id: str) -> Decimal:
        """Calculate total spend across all channels for today"""
        total = Decimal('0')
        if campaign_id in self.channel_budgets:
            for channel in self.channel_budgets[campaign_id]:
                total += self._calculate_today_spend(campaign_id, channel)
        return total
    
    def _get_time_elapsed_pct(self) -> float:
        """Get percentage of day elapsed"""
        current_time = datetime.utcnow()
        start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed = current_time - start_of_day
        return elapsed.total_seconds() / (24 * 3600)


# Example usage and testing
if __name__ == "__main__":
    async def test_budget_pacer():
        """Test the budget pacer functionality"""
        
        # Initialize pacer
        pacer = BudgetPacer()
        
        # Set up a campaign
        campaign_id = "test_campaign_001"
        daily_budget = Decimal('1000.00')
        
        # Allocate budget for Google Ads
        allocations = pacer.allocate_hourly_budget(
            campaign_id, 
            ChannelType.GOOGLE_ADS, 
            daily_budget,
            PacingStrategy.ADAPTIVE_HYBRID
        )
        
        print(f"Hourly allocations for {campaign_id}:")
        for allocation in allocations[:6]:  # Show first 6 hours
            print(f"  Hour {allocation.hour}: {allocation.base_allocation_pct:.1%} "
                  f"(${float(daily_budget * Decimal(str(allocation.base_allocation_pct))):.2f})")
        
        # Test spend authorization
        test_spend = Decimal('50.00')
        can_spend, reason = pacer.can_spend(campaign_id, ChannelType.GOOGLE_ADS, test_spend)
        print(f"\nCan spend ${test_spend}: {can_spend} - {reason}")
        
        # Record some transactions
        transaction = SpendTransaction(
            campaign_id=campaign_id,
            channel=ChannelType.GOOGLE_ADS,
            amount=test_spend,
            timestamp=datetime.utcnow(),
            clicks=25,
            conversions=2,
            cost_per_click=2.00,
            conversion_rate=0.08
        )
        
        pacer.record_spend(transaction)
        
        # Check pacing
        pace_ratio, alert = await pacer.check_pace(campaign_id, ChannelType.GOOGLE_ADS)
        print(f"\nCurrent pace ratio: {pace_ratio:.2f}")
        if alert:
            print(f"Alert: {alert.alert_type} - {alert.recommended_action}")
        
        print("\nBudget pacer test completed successfully!")
    
    # Run the test
    asyncio.run(test_budget_pacer())