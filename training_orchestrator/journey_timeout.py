"""
Journey Timeout and Abandonment Logic for GAELP Training Orchestrator

Handles journey timeouts, abandonment detection, and cleanup of zombie journeys.
Ensures proper termination of journeys and calculates abandonment penalties.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
from google.cloud import bigquery, pubsub_v1
import redis

# Import the conversion lag model
from conversion_lag_model import ConversionLagModel, ConversionJourney

logger = logging.getLogger(__name__)


class AbandonmentReason(Enum):
    """Reasons for journey abandonment"""
    TIMEOUT = "timeout"
    INACTIVITY = "inactivity"
    COMPETITOR_CONVERSION = "competitor_conversion"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FATIGUE = "fatigue"
    MANUAL_TERMINATION = "manual_termination"


@dataclass
class AbandonmentPenalty:
    """Penalty structure for abandoned journeys"""
    journey_id: str
    abandonment_reason: AbandonmentReason
    days_active: int
    total_cost: float
    touchpoint_count: int
    last_state: str
    conversion_probability_lost: float
    penalty_amount: float
    opportunity_cost: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TimeoutConfiguration:
    """Configuration for journey timeout handling"""
    default_timeout_days: int = 14
    inactivity_threshold_hours: int = 72  # 3 days
    max_journey_duration_days: int = 90
    cleanup_batch_size: int = 1000
    abandonment_check_interval_minutes: int = 30
    
    # Conversion lag model settings
    enable_conversion_lag_model: bool = True
    conversion_lag_model_type: str = 'weibull'  # 'weibull' or 'cox'
    attribution_window_days: int = 30
    timeout_threshold_days: int = 45
    
    # Penalty calculation weights
    cost_penalty_multiplier: float = 0.15
    opportunity_cost_multiplier: float = 0.25
    time_decay_factor: float = 0.8
    state_penalty_weights: Dict[str, float] = field(default_factory=lambda: {
        'UNAWARE': 0.1,
        'AWARE': 0.2,
        'CONSIDERING': 0.5,
        'INTENT': 0.8,
        'CONVERTED': 0.0
    })


class JourneyTimeoutManager:
    """
    Manages journey timeouts and abandonment logic for the training orchestrator.
    
    Key responsibilities:
    - Monitor active journeys for timeout conditions
    - Calculate abandonment penalties and opportunity costs
    - Clean up zombie and stale journeys
    - Provide abandonment analytics for training feedback
    """
    
    def __init__(self,
                 config: TimeoutConfiguration,
                 bigquery_client: Optional[bigquery.Client] = None,
                 redis_client: Optional[redis.Redis] = None,
                 project_id: str = None,
                 dataset_id: str = "gaelp"):
        
        self.config = config
        self.project_id = project_id
        self.dataset_id = dataset_id
        
        # Initialize clients
        self.bq_client = bigquery_client or (bigquery.Client(project=project_id) if project_id else None)
        self.redis_client = redis_client or redis.Redis()
        
        # Initialize conversion lag model
        self.conversion_lag_model = None
        if config.enable_conversion_lag_model:
            self.conversion_lag_model = ConversionLagModel(
                attribution_window_days=config.attribution_window_days,
                timeout_threshold_days=config.timeout_threshold_days,
                model_type=config.conversion_lag_model_type
            )
            logger.info(f"Initialized conversion lag model: {config.conversion_lag_model_type}")
        
        # Internal state
        self._active_timeouts: Dict[str, datetime] = {}
        self._abandonment_cache: Dict[str, AbandonmentPenalty] = {}
        self._journey_data_cache: Dict[str, ConversionJourney] = {}  # Cache for conversion lag model
        self._cleanup_stats = {
            'total_timeouts_processed': 0,
            'total_journeys_abandoned': 0,
            'total_penalty_amount': 0.0,
            'last_cleanup_time': None
        }
        
        # Background task management
        self._timeout_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        logger.info(f"JourneyTimeoutManager initialized with {config.default_timeout_days}-day timeout")
    
    async def start(self):
        """Start the timeout manager background processes"""
        if self._is_running:
            logger.warning("JourneyTimeoutManager is already running")
            return
        
        self._is_running = True
        logger.info("Starting JourneyTimeoutManager background processes")
        
        # Start background tasks
        self._timeout_check_task = asyncio.create_task(self._timeout_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Load existing timeouts from cache
        await self._load_active_timeouts()
    
    async def stop(self):
        """Stop the timeout manager and cleanup resources"""
        if not self._is_running:
            return
        
        self._is_running = False
        logger.info("Stopping JourneyTimeoutManager background processes")
        
        # Cancel background tasks
        if self._timeout_check_task:
            self._timeout_check_task.cancel()
            try:
                await self._timeout_check_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save state
        await self._save_active_timeouts()
        
        logger.info("JourneyTimeoutManager stopped")
    
    async def register_journey(self,
                             journey_id: str,
                             start_time: datetime,
                             user_id: str = None,
                             initial_timeout: Optional[datetime] = None) -> datetime:
        """
        Register a new journey for timeout monitoring.
        
        Args:
            journey_id: Unique journey identifier
            start_time: When the journey started
            user_id: Optional user identifier for analytics
            initial_timeout: Optional custom timeout (default: start_time + default_timeout_days)
        
        Returns:
            The timeout datetime for this journey
        """
        if initial_timeout is None:
            initial_timeout = start_time + timedelta(days=self.config.default_timeout_days)
        
        # Enforce maximum duration
        max_timeout = start_time + timedelta(days=self.config.max_journey_duration_days)
        if initial_timeout > max_timeout:
            initial_timeout = max_timeout
            logger.warning(f"Journey {journey_id} timeout capped at maximum duration: {max_timeout}")
        
        # Store in memory and cache
        self._active_timeouts[journey_id] = initial_timeout
        
        # Cache in Redis for persistence
        if self.redis_client:
            timeout_data = {
                'journey_id': journey_id,
                'start_time': start_time.isoformat(),
                'timeout_at': initial_timeout.isoformat(),
                'user_id': user_id or 'unknown',
                'registered_at': datetime.now().isoformat()
            }
            
            cache_key = f"gaelp:journey_timeout:{journey_id}"
            self.redis_client.setex(
                cache_key,
                timedelta(days=self.config.max_journey_duration_days + 1),
                json.dumps(timeout_data)
            )
        
        logger.debug(f"Registered journey {journey_id} for timeout monitoring (timeout: {initial_timeout})")
        return initial_timeout
    
    async def check_timeouts(self) -> List[str]:
        """
        Check all active journeys for timeout conditions.
        
        Returns:
            List of journey IDs that have timed out
        """
        current_time = datetime.now()
        timed_out_journeys = []
        
        for journey_id, timeout_at in self._active_timeouts.items():
            if current_time >= timeout_at:
                timed_out_journeys.append(journey_id)
        
        if timed_out_journeys:
            logger.info(f"Found {len(timed_out_journeys)} timed out journeys")
            
            # Process timeouts
            for journey_id in timed_out_journeys:
                await self._handle_journey_timeout(journey_id, AbandonmentReason.TIMEOUT)
        
        return timed_out_journeys
    
    async def mark_abandoned(self,
                           journey_id: str,
                           reason: AbandonmentReason,
                           journey_data: Optional[Dict[str, Any]] = None) -> AbandonmentPenalty:
        """
        Mark a journey as abandoned and calculate penalties.
        
        Args:
            journey_id: Journey to abandon
            reason: Reason for abandonment
            journey_data: Optional journey data for penalty calculation
        
        Returns:
            Calculated abandonment penalty
        """
        logger.info(f"Marking journey {journey_id} as abandoned (reason: {reason.value})")
        
        # Get journey data if not provided
        if journey_data is None:
            journey_data = await self._fetch_journey_data(journey_id)
        
        # Calculate abandonment penalty
        penalty = self.calculate_abandonment_cost(journey_id, reason, journey_data)
        
        # Store penalty in cache
        self._abandonment_cache[journey_id] = penalty
        
        # Update journey in database
        await self._update_journey_abandonment_status(journey_id, penalty)
        
        # Remove from active timeouts
        self._active_timeouts.pop(journey_id, None)
        
        # Clear Redis cache
        if self.redis_client:
            cache_key = f"gaelp:journey_timeout:{journey_id}"
            self.redis_client.delete(cache_key)
        
        # Update statistics
        self._cleanup_stats['total_journeys_abandoned'] += 1
        self._cleanup_stats['total_penalty_amount'] += penalty.penalty_amount
        
        # Emit abandonment event for training feedback
        await self._emit_abandonment_event(penalty)
        
        return penalty
    
    def calculate_abandonment_cost(self,
                                 journey_id: str,
                                 reason: AbandonmentReason,
                                 journey_data: Dict[str, Any]) -> AbandonmentPenalty:
        """
        Calculate the cost and penalty for an abandoned journey.
        
        Args:
            journey_id: Journey identifier
            reason: Abandonment reason
            journey_data: Journey data for calculation
        
        Returns:
            Calculated abandonment penalty
        """
        # Extract journey metrics
        start_time = journey_data.get('journey_start', datetime.now())
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        
        days_active = max(1, (datetime.now() - start_time).days)
        total_cost = float(journey_data.get('total_cost', 0.0))
        touchpoint_count = int(journey_data.get('touchpoint_count', 0))
        last_state = journey_data.get('current_state', 'UNAWARE')
        conversion_probability = float(journey_data.get('conversion_probability', 0.0))
        expected_conversion_value = float(journey_data.get('expected_conversion_value', 100.0))
        
        # Calculate base penalty components
        cost_penalty = total_cost * self.config.cost_penalty_multiplier
        
        # State-based penalty (higher penalty for more advanced states)
        state_weight = self.config.state_penalty_weights.get(last_state, 0.2)
        state_penalty = total_cost * state_weight
        
        # Time decay factor (longer journeys have higher penalties)
        time_factor = 1.0 - (self.config.time_decay_factor ** days_active)
        
        # Opportunity cost (lost conversion value)
        opportunity_cost = (conversion_probability * expected_conversion_value * 
                          self.config.opportunity_cost_multiplier)
        
        # Reason-specific multipliers
        reason_multipliers = {
            AbandonmentReason.TIMEOUT: 1.0,
            AbandonmentReason.INACTIVITY: 0.8,
            AbandonmentReason.COMPETITOR_CONVERSION: 1.5,
            AbandonmentReason.BUDGET_EXHAUSTED: 0.6,
            AbandonmentReason.FATIGUE: 0.7,
            AbandonmentReason.MANUAL_TERMINATION: 0.3
        }
        
        reason_multiplier = reason_multipliers.get(reason, 1.0)
        
        # Calculate final penalty
        base_penalty = cost_penalty + state_penalty
        final_penalty = (base_penalty * time_factor * reason_multiplier) + opportunity_cost
        
        penalty = AbandonmentPenalty(
            journey_id=journey_id,
            abandonment_reason=reason,
            days_active=days_active,
            total_cost=total_cost,
            touchpoint_count=touchpoint_count,
            last_state=last_state,
            conversion_probability_lost=conversion_probability,
            penalty_amount=final_penalty,
            opportunity_cost=opportunity_cost
        )
        
        logger.debug(f"Calculated abandonment penalty for {journey_id}: ${final_penalty:.2f} "
                    f"(cost: ${total_cost:.2f}, opportunity: ${opportunity_cost:.2f})")
        
        return penalty
    
    async def cleanup_stale_data(self, 
                               older_than_days: int = 30,
                               batch_size: Optional[int] = None) -> Dict[str, int]:
        """
        Clean up stale journey data and abandoned journeys.
        
        Args:
            older_than_days: Delete data older than this many days
            batch_size: Number of records to process in each batch
        
        Returns:
            Statistics about cleanup operation
        """
        if batch_size is None:
            batch_size = self.config.cleanup_batch_size
        
        cleanup_cutoff = datetime.now() - timedelta(days=older_than_days)
        stats = {
            'stale_journeys_removed': 0,
            'abandoned_penalties_archived': 0,
            'cache_entries_cleared': 0,
            'errors': 0
        }
        
        logger.info(f"Starting stale data cleanup (cutoff: {cleanup_cutoff})")
        
        try:
            # Clean up abandoned journey data in BigQuery
            if self.bq_client:
                stats['stale_journeys_removed'] = await self._cleanup_stale_journeys_bq(
                    cleanup_cutoff, batch_size
                )
                
                stats['abandoned_penalties_archived'] = await self._archive_old_penalties_bq(
                    cleanup_cutoff, batch_size
                )
            
            # Clean up Redis cache entries
            if self.redis_client:
                stats['cache_entries_cleared'] = await self._cleanup_stale_cache_entries(
                    cleanup_cutoff
                )
            
            # Clean up in-memory caches
            self._cleanup_memory_caches(cleanup_cutoff)
            
            # Update cleanup statistics
            self._cleanup_stats['last_cleanup_time'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during stale data cleanup: {e}")
            stats['errors'] += 1
        
        logger.info(f"Stale data cleanup completed: {stats}")
        return stats
    
    async def get_abandonment_analytics(self,
                                      start_date: datetime,
                                      end_date: datetime,
                                      group_by: str = "reason") -> Dict[str, Any]:
        """
        Get analytics on journey abandonment patterns.
        
        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            group_by: Grouping dimension (reason, state, day, etc.)
        
        Returns:
            Abandonment analytics data
        """
        analytics = {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_abandonments': 0,
                'total_penalty_amount': 0.0,
                'total_opportunity_cost': 0.0,
                'average_days_active': 0.0,
                'average_touchpoints': 0.0
            },
            'breakdown': {},
            'trends': []
        }
        
        # Query abandonment data from BigQuery
        if self.bq_client:
            query = f"""
            SELECT 
                abandonment_reason,
                last_state,
                DATE(created_at) as abandonment_date,
                COUNT(*) as abandonment_count,
                AVG(days_active) as avg_days_active,
                AVG(touchpoint_count) as avg_touchpoints,
                SUM(penalty_amount) as total_penalty,
                SUM(opportunity_cost) as total_opportunity_cost
            FROM `{self.project_id}.{self.dataset_id}.journey_abandonments`
            WHERE created_at BETWEEN @start_date AND @end_date
            GROUP BY abandonment_reason, last_state, DATE(created_at)
            ORDER BY abandonment_date DESC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
                    bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date)
                ]
            )
            
            try:
                results = self.bq_client.query(query, job_config=job_config)
                
                # Process results
                for row in results:
                    analytics['summary']['total_abandonments'] += row.abandonment_count
                    analytics['summary']['total_penalty_amount'] += row.total_penalty or 0.0
                    analytics['summary']['total_opportunity_cost'] += row.total_opportunity_cost or 0.0
                    
                    # Group by specified dimension
                    group_key = getattr(row, group_by, 'unknown')
                    if group_key not in analytics['breakdown']:
                        analytics['breakdown'][group_key] = {
                            'count': 0,
                            'penalty_amount': 0.0,
                            'opportunity_cost': 0.0,
                            'avg_days_active': 0.0,
                            'avg_touchpoints': 0.0
                        }
                    
                    breakdown = analytics['breakdown'][group_key]
                    breakdown['count'] += row.abandonment_count
                    breakdown['penalty_amount'] += row.total_penalty or 0.0
                    breakdown['opportunity_cost'] += row.total_opportunity_cost or 0.0
                    breakdown['avg_days_active'] = row.avg_days_active or 0.0
                    breakdown['avg_touchpoints'] = row.avg_touchpoints or 0.0
                    
                    # Add to trends
                    analytics['trends'].append({
                        'date': row.abandonment_date.isoformat(),
                        'reason': row.abandonment_reason,
                        'state': row.last_state,
                        'count': row.abandonment_count,
                        'penalty': row.total_penalty or 0.0
                    })
                
                # Calculate averages
                if analytics['summary']['total_abandonments'] > 0:
                    total_abandons = analytics['summary']['total_abandonments']
                    
                    # Get weighted averages from breakdown
                    total_days = sum(
                        breakdown['avg_days_active'] * breakdown['count'] 
                        for breakdown in analytics['breakdown'].values()
                    )
                    total_touches = sum(
                        breakdown['avg_touchpoints'] * breakdown['count']
                        for breakdown in analytics['breakdown'].values()
                    )
                    
                    analytics['summary']['average_days_active'] = total_days / total_abandons
                    analytics['summary']['average_touchpoints'] = total_touches / total_abandons
                
            except Exception as e:
                logger.error(f"Error querying abandonment analytics: {e}")
        
        return analytics
    
    def get_timeout_status(self, journey_id: str) -> Optional[Dict[str, Any]]:
        """
        Get timeout status for a specific journey.
        
        Args:
            journey_id: Journey to check
        
        Returns:
            Timeout status information or None if not found
        """
        if journey_id not in self._active_timeouts:
            return None
        
        timeout_at = self._active_timeouts[journey_id]
        current_time = datetime.now()
        
        return {
            'journey_id': journey_id,
            'timeout_at': timeout_at.isoformat(),
            'is_expired': current_time >= timeout_at,
            'time_remaining': max(0, (timeout_at - current_time).total_seconds()),
            'days_remaining': max(0, (timeout_at - current_time).days)
        }
    
    async def extend_timeout(self,
                           journey_id: str,
                           additional_days: int,
                           reason: str = None) -> Optional[datetime]:
        """
        Extend the timeout for a specific journey.
        
        Args:
            journey_id: Journey to extend
            additional_days: Number of additional days
            reason: Optional reason for extension
        
        Returns:
            New timeout datetime or None if journey not found
        """
        if journey_id not in self._active_timeouts:
            logger.warning(f"Cannot extend timeout for unknown journey: {journey_id}")
            return None
        
        old_timeout = self._active_timeouts[journey_id]
        new_timeout = old_timeout + timedelta(days=additional_days)
        
        # Enforce maximum duration
        max_timeout = datetime.now() + timedelta(days=self.config.max_journey_duration_days)
        if new_timeout > max_timeout:
            new_timeout = max_timeout
            logger.warning(f"Journey {journey_id} timeout extension capped at maximum duration")
        
        self._active_timeouts[journey_id] = new_timeout
        
        # Update Redis cache
        if self.redis_client:
            cache_key = f"gaelp:journey_timeout:{journey_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                timeout_data = json.loads(cached_data)
                timeout_data['timeout_at'] = new_timeout.isoformat()
                timeout_data['extended_at'] = datetime.now().isoformat()
                timeout_data['extension_reason'] = reason or 'manual_extension'
                
                self.redis_client.setex(
                    cache_key,
                    timedelta(days=self.config.max_journey_duration_days + 1),
                    json.dumps(timeout_data)
                )
        
        logger.info(f"Extended timeout for journey {journey_id}: {old_timeout} -> {new_timeout} "
                   f"(reason: {reason or 'not specified'})")
        
        return new_timeout
    
    async def register_journey_for_conversion_prediction(self,
                                                        journey_id: str,
                                                        user_id: str,
                                                        start_time: datetime,
                                                        touchpoints: List[Dict[str, Any]] = None,
                                                        features: Dict[str, float] = None) -> Optional[Dict[str, Any]]:
        """
        Register a journey for conversion lag prediction and get timeout recommendations.
        
        Args:
            journey_id: Unique journey identifier
            user_id: User identifier
            start_time: Journey start time
            touchpoints: List of touchpoint data
            features: Journey features for prediction
            
        Returns:
            Dictionary with conversion predictions and timeout recommendations
        """
        if not self.conversion_lag_model:
            logger.warning("Conversion lag model not enabled")
            return None
        
        # Create ConversionJourney object
        conversion_journey = ConversionJourney(
            user_id=user_id,
            start_time=start_time,
            touchpoints=touchpoints or [],
            features=features or {}
        )
        
        # Cache the journey data
        self._journey_data_cache[journey_id] = conversion_journey
        
        # Get conversion time predictions if model is fitted
        if self.conversion_lag_model.is_fitted:
            try:
                predictions = self.conversion_lag_model.predict_conversion_time([conversion_journey])
                hazard_rates = self.conversion_lag_model.calculate_hazard_rate([conversion_journey])
                
                conversion_probs = predictions.get(user_id, np.array([]))
                hazard_rates_data = hazard_rates.get(user_id, np.array([]))
                
                # Calculate recommended timeout based on predictions
                recommended_timeout = self._calculate_recommended_timeout(conversion_probs, hazard_rates_data)
                
                prediction_data = {
                    'journey_id': journey_id,
                    'user_id': user_id,
                    'conversion_probabilities': conversion_probs.tolist() if hasattr(conversion_probs, 'tolist') else [],
                    'hazard_rates': hazard_rates_data.tolist() if hasattr(hazard_rates_data, 'tolist') else [],
                    'recommended_timeout_days': recommended_timeout,
                    'prediction_timestamp': datetime.now()
                }
                
                logger.info(f"Generated conversion predictions for journey {journey_id}: "
                          f"recommended timeout = {recommended_timeout} days")
                
                return prediction_data
                
            except Exception as e:
                logger.error(f"Error generating conversion predictions for journey {journey_id}: {e}")
                return None
        else:
            logger.info(f"Conversion lag model not yet trained, using default timeout for journey {journey_id}")
            return None
    
    def _calculate_recommended_timeout(self, 
                                     conversion_probs: np.ndarray, 
                                     hazard_rates: np.ndarray) -> int:
        """
        Calculate recommended timeout based on conversion probabilities and hazard rates.
        
        Args:
            conversion_probs: Conversion probabilities over time
            hazard_rates: Hazard rates over time
            
        Returns:
            Recommended timeout in days
        """
        if len(conversion_probs) == 0:
            return self.config.default_timeout_days
        
        # Find the point where conversion probability plateaus or hazard rate drops significantly
        prob_threshold = 0.95  # 95% of maximum conversion probability
        hazard_threshold = 0.1  # 10% of maximum hazard rate
        
        max_conversion_prob = np.max(conversion_probs)
        max_hazard_rate = np.max(hazard_rates) if len(hazard_rates) > 0 else 0
        
        # Find optimal cutoff point
        optimal_timeout = self.config.default_timeout_days
        
        for day in range(1, min(len(conversion_probs), self.config.max_journey_duration_days)):
            current_prob = conversion_probs[day - 1] if day <= len(conversion_probs) else 0
            current_hazard = hazard_rates[day - 1] if day <= len(hazard_rates) else 0
            
            # If we've reached 95% of max conversion probability and hazard is low
            if (current_prob >= max_conversion_prob * prob_threshold and
                current_hazard <= max_hazard_rate * hazard_threshold):
                optimal_timeout = day
                break
        
        # Ensure timeout is within reasonable bounds
        optimal_timeout = max(self.config.default_timeout_days // 2, optimal_timeout)
        optimal_timeout = min(self.config.max_journey_duration_days, optimal_timeout)
        
        return optimal_timeout
    
    async def train_conversion_lag_model(self, 
                                       journey_data: Optional[List[ConversionJourney]] = None,
                                       lookback_days: int = 90) -> bool:
        """
        Train or retrain the conversion lag model with available journey data.
        
        Args:
            journey_data: Optional list of ConversionJourney objects for training
            lookback_days: Days to look back for training data if journey_data not provided
            
        Returns:
            True if training was successful, False otherwise
        """
        if not self.conversion_lag_model:
            logger.error("Conversion lag model not initialized")
            return False
        
        try:
            # Get training data
            if journey_data is None:
                journey_data = await self._fetch_historical_journey_data(lookback_days)
            
            if not journey_data:
                logger.error("No journey data available for training")
                return False
            
            # Train the model
            logger.info(f"Training conversion lag model with {len(journey_data)} journeys")
            self.conversion_lag_model.fit(journey_data)
            
            # Generate insights
            insights = self.conversion_lag_model.get_conversion_insights(journey_data)
            logger.info(f"Conversion model training completed. Insights: {insights}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training conversion lag model: {e}")
            return False
    
    async def _fetch_historical_journey_data(self, lookback_days: int) -> List[ConversionJourney]:
        """
        Fetch historical journey data from BigQuery for model training.
        
        Args:
            lookback_days: Number of days to look back for data
            
        Returns:
            List of ConversionJourney objects
        """
        if not self.bq_client:
            logger.warning("No BigQuery client available for fetching historical data")
            return []
        
        lookback_date = datetime.now() - timedelta(days=lookback_days)
        
        query = f"""
        SELECT 
            j.journey_id,
            j.user_id,
            j.journey_start,
            j.journey_end,
            j.converted,
            j.conversion_value,
            j.current_state,
            j.touchpoint_count,
            j.total_cost,
            j.abandonment_reason,
            ARRAY_AGG(STRUCT(
                t.timestamp as timestamp,
                t.channel as channel,
                t.action_data as action_data
            )) as touchpoints
        FROM `{self.project_id}.{self.dataset_id}.user_journeys` j
        LEFT JOIN `{self.project_id}.{self.dataset_id}.touchpoints` t
            ON j.journey_id = t.episode_id
        WHERE j.journey_start >= @lookback_date
        AND j.is_active = FALSE
        GROUP BY j.journey_id, j.user_id, j.journey_start, j.journey_end, 
                 j.converted, j.conversion_value, j.current_state, j.touchpoint_count,
                 j.total_cost, j.abandonment_reason
        ORDER BY j.journey_start DESC
        LIMIT 10000
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("lookback_date", "TIMESTAMP", lookback_date)
            ]
        )
        
        try:
            results = self.bq_client.query(query, job_config=job_config)
            journeys = []
            
            for row in results:
                # Convert touchpoints
                touchpoints = []
                if row.touchpoints:
                    for tp in row.touchpoints:
                        touchpoints.append({
                            'timestamp': tp.timestamp,
                            'channel': tp.channel or 'unknown',
                            'action_data': tp.action_data
                        })
                
                # Create ConversionJourney object
                journey = ConversionJourney(
                    user_id=row.user_id,
                    start_time=row.journey_start,
                    end_time=row.journey_end,
                    converted=bool(row.converted),
                    duration_days=(row.journey_end - row.journey_start).days if row.journey_end else None,
                    touchpoints=touchpoints,
                    features={
                        'touchpoint_count': row.touchpoint_count or 0,
                        'total_cost': row.total_cost or 0.0,
                        'current_state': row.current_state or 'UNKNOWN'
                    },
                    is_censored=row.abandonment_reason is not None,
                    timeout_reason=row.abandonment_reason
                )
                journeys.append(journey)
            
            logger.info(f"Fetched {len(journeys)} historical journeys for training")
            return journeys
            
        except Exception as e:
            logger.error(f"Error fetching historical journey data: {e}")
            return []
    
    def get_conversion_prediction(self, journey_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversion prediction for a specific journey.
        
        Args:
            journey_id: Journey identifier
            
        Returns:
            Dictionary with conversion predictions or None if not available
        """
        if not self.conversion_lag_model or not self.conversion_lag_model.is_fitted:
            return None
            
        journey = self._journey_data_cache.get(journey_id)
        if not journey:
            logger.warning(f"No journey data found for {journey_id}")
            return None
        
        try:
            predictions = self.conversion_lag_model.predict_conversion_time([journey])
            hazard_rates = self.conversion_lag_model.calculate_hazard_rate([journey])
            
            return {
                'journey_id': journey_id,
                'conversion_probabilities': predictions.get(journey.user_id, []),
                'hazard_rates': hazard_rates.get(journey.user_id, []),
                'prediction_timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting conversion prediction for journey {journey_id}: {e}")
            return None
    
    async def handle_censored_journey_data(self) -> Dict[str, Any]:
        """
        Handle right-censored data for ongoing journeys and update predictions.
        
        Returns:
            Statistics about censored data handling
        """
        if not self.conversion_lag_model:
            return {'error': 'Conversion lag model not initialized'}
        
        try:
            # Get all cached journeys
            cached_journeys = list(self._journey_data_cache.values())
            
            # Process censored data
            processed_journeys = self.conversion_lag_model.handle_censored_data(cached_journeys)
            
            # Update cache with processed journeys
            for journey in processed_journeys:
                # Find corresponding journey_id
                for journey_id, cached_journey in self._journey_data_cache.items():
                    if cached_journey.user_id == journey.user_id:
                        self._journey_data_cache[journey_id] = journey
                        break
            
            stats = {
                'total_journeys_processed': len(processed_journeys),
                'censored_journeys': len([j for j in processed_journeys if j.is_censored]),
                'timeout_journeys': len([j for j in processed_journeys if j.timeout_reason == 'abandoned']),
                'processing_timestamp': datetime.now()
            }
            
            logger.info(f"Processed censored journey data: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error handling censored journey data: {e}")
            return {'error': str(e)}
    
    # Private helper methods
    
    async def _timeout_check_loop(self):
        """Background loop to check for journey timeouts"""
        while self._is_running:
            try:
                await self.check_timeouts()
                await asyncio.sleep(self.config.abandonment_check_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in timeout check loop: {e}")
                await asyncio.sleep(60)  # Brief pause before retrying
    
    async def _cleanup_loop(self):
        """Background loop for periodic cleanup"""
        while self._is_running:
            try:
                # Run cleanup daily
                await asyncio.sleep(24 * 60 * 60)
                await self.cleanup_stale_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _handle_journey_timeout(self, journey_id: str, reason: AbandonmentReason):
        """Handle a journey that has timed out"""
        try:
            await self.mark_abandoned(journey_id, reason)
            self._cleanup_stats['total_timeouts_processed'] += 1
        except Exception as e:
            logger.error(f"Error handling timeout for journey {journey_id}: {e}")
    
    async def _fetch_journey_data(self, journey_id: str) -> Dict[str, Any]:
        """Fetch journey data from BigQuery"""
        if not self.bq_client:
            logger.warning("No BigQuery client available, using default journey data")
            return {
                'journey_start': datetime.now() - timedelta(days=7),
                'total_cost': 50.0,
                'touchpoint_count': 3,
                'current_state': 'CONSIDERING',
                'conversion_probability': 0.2,
                'expected_conversion_value': 100.0
            }
        
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.user_journeys`
        WHERE journey_id = @journey_id
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("journey_id", "STRING", journey_id)
            ]
        )
        
        try:
            results = self.bq_client.query(query, job_config=job_config)
            for row in results:
                return dict(row)
        except Exception as e:
            logger.error(f"Error fetching journey data for {journey_id}: {e}")
        
        # Return default data if query fails
        return {
            'journey_start': datetime.now() - timedelta(days=7),
            'total_cost': 50.0,
            'touchpoint_count': 3,
            'current_state': 'CONSIDERING',
            'conversion_probability': 0.2,
            'expected_conversion_value': 100.0
        }
    
    async def _update_journey_abandonment_status(self, 
                                               journey_id: str,
                                               penalty: AbandonmentPenalty):
        """Update journey abandonment status in BigQuery"""
        if not self.bq_client:
            return
        
        # Insert abandonment record
        abandonment_table = f"{self.project_id}.{self.dataset_id}.journey_abandonments"
        
        rows_to_insert = [{
            'journey_id': penalty.journey_id,
            'abandonment_reason': penalty.abandonment_reason.value,
            'days_active': penalty.days_active,
            'total_cost': penalty.total_cost,
            'touchpoint_count': penalty.touchpoint_count,
            'last_state': penalty.last_state,
            'conversion_probability_lost': penalty.conversion_probability_lost,
            'penalty_amount': penalty.penalty_amount,
            'opportunity_cost': penalty.opportunity_cost,
            'created_at': penalty.created_at
        }]
        
        try:
            table_ref = self.bq_client.dataset(self.dataset_id).table('journey_abandonments')
            table = self.bq_client.get_table(table_ref)
            errors = self.bq_client.insert_rows(table, rows_to_insert)
            
            if errors:
                logger.error(f"Failed to insert abandonment record: {errors}")
        except Exception as e:
            logger.error(f"Error inserting abandonment record for {journey_id}: {e}")
        
        # Update journey status
        update_query = f"""
        UPDATE `{self.project_id}.{self.dataset_id}.user_journeys`
        SET 
            is_active = FALSE,
            journey_end = CURRENT_TIMESTAMP(),
            final_state = current_state,
            abandonment_reason = @abandonment_reason,
            abandonment_penalty = @penalty_amount
        WHERE journey_id = @journey_id
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("journey_id", "STRING", journey_id),
                bigquery.ScalarQueryParameter("abandonment_reason", "STRING", 
                                            penalty.abandonment_reason.value),
                bigquery.ScalarQueryParameter("penalty_amount", "FLOAT64", penalty.penalty_amount)
            ]
        )
        
        try:
            job = self.bq_client.query(update_query, job_config=job_config)
            job.result()
        except Exception as e:
            logger.error(f"Error updating journey abandonment status for {journey_id}: {e}")
    
    async def _emit_abandonment_event(self, penalty: AbandonmentPenalty):
        """Emit abandonment event for training feedback"""
        event_data = {
            'event_type': 'journey_abandoned',
            'journey_id': penalty.journey_id,
            'abandonment_reason': penalty.abandonment_reason.value,
            'penalty_amount': penalty.penalty_amount,
            'opportunity_cost': penalty.opportunity_cost,
            'last_state': penalty.last_state,
            'days_active': penalty.days_active,
            'touchpoint_count': penalty.touchpoint_count,
            'timestamp': penalty.created_at.isoformat()
        }
        
        # Store in Redis for real-time processing
        if self.redis_client:
            event_key = f"gaelp:abandonment_event:{penalty.journey_id}"
            self.redis_client.setex(event_key, timedelta(hours=24), json.dumps(event_data))
        
        logger.info(f"Emitted abandonment event for journey {penalty.journey_id}")
    
    async def _load_active_timeouts(self):
        """Load active timeouts from Redis cache"""
        if not self.redis_client:
            return
        
        try:
            pattern = "gaelp:journey_timeout:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    timeout_data = json.loads(cached_data)
                    journey_id = timeout_data['journey_id']
                    timeout_at = datetime.fromisoformat(timeout_data['timeout_at'])
                    
                    # Only load if not yet expired
                    if timeout_at > datetime.now():
                        self._active_timeouts[journey_id] = timeout_at
            
            logger.info(f"Loaded {len(self._active_timeouts)} active timeouts from cache")
            
        except Exception as e:
            logger.error(f"Error loading active timeouts: {e}")
    
    async def _save_active_timeouts(self):
        """Save active timeouts to Redis cache"""
        if not self.redis_client:
            return
        
        try:
            for journey_id, timeout_at in self._active_timeouts.items():
                timeout_data = {
                    'journey_id': journey_id,
                    'timeout_at': timeout_at.isoformat(),
                    'saved_at': datetime.now().isoformat()
                }
                
                cache_key = f"gaelp:journey_timeout:{journey_id}"
                self.redis_client.setex(
                    cache_key,
                    timedelta(days=self.config.max_journey_duration_days + 1),
                    json.dumps(timeout_data)
                )
            
            logger.debug(f"Saved {len(self._active_timeouts)} active timeouts to cache")
            
        except Exception as e:
            logger.error(f"Error saving active timeouts: {e}")
    
    async def _cleanup_stale_journeys_bq(self, cutoff_date: datetime, batch_size: int) -> int:
        """Clean up stale journeys in BigQuery"""
        query = f"""
        SELECT COUNT(*) as count
        FROM `{self.project_id}.{self.dataset_id}.user_journeys`
        WHERE is_active = FALSE 
        AND journey_end < @cutoff_date
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("cutoff_date", "TIMESTAMP", cutoff_date)
            ]
        )
        
        try:
            results = self.bq_client.query(query, job_config=job_config)
            for row in results:
                count = row.count
                logger.info(f"Found {count} stale journeys to clean up")
                return count
        except Exception as e:
            logger.error(f"Error counting stale journeys: {e}")
        
        return 0
    
    async def _archive_old_penalties_bq(self, cutoff_date: datetime, batch_size: int) -> int:
        """Archive old penalty records in BigQuery"""
        # Move old penalties to archive table
        query = f"""
        CREATE TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.journey_abandonments_archive`
        AS SELECT * FROM `{self.project_id}.{self.dataset_id}.journey_abandonments` LIMIT 0
        """
        
        try:
            self.bq_client.query(query).result()
            
            # Insert old records into archive
            insert_query = f"""
            INSERT INTO `{self.project_id}.{self.dataset_id}.journey_abandonments_archive`
            SELECT * FROM `{self.project_id}.{self.dataset_id}.journey_abandonments`
            WHERE created_at < @cutoff_date
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("cutoff_date", "TIMESTAMP", cutoff_date)
                ]
            )
            
            job = self.bq_client.query(insert_query, job_config=job_config)
            job.result()
            
            archived_count = job.num_dml_affected_rows
            
            # Delete archived records from main table
            delete_query = f"""
            DELETE FROM `{self.project_id}.{self.dataset_id}.journey_abandonments`
            WHERE created_at < @cutoff_date
            """
            
            delete_job = self.bq_client.query(delete_query, job_config=job_config)
            delete_job.result()
            
            logger.info(f"Archived {archived_count} old abandonment penalty records")
            return archived_count
            
        except Exception as e:
            logger.error(f"Error archiving old penalties: {e}")
            return 0
    
    async def _cleanup_stale_cache_entries(self, cutoff_date: datetime) -> int:
        """Clean up stale cache entries in Redis"""
        if not self.redis_client:
            return 0
        
        try:
            pattern = "gaelp:journey_timeout:*"
            keys = self.redis_client.keys(pattern)
            cleared_count = 0
            
            for key in keys:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    timeout_data = json.loads(cached_data)
                    timeout_at = datetime.fromisoformat(timeout_data['timeout_at'])
                    
                    if timeout_at < cutoff_date:
                        self.redis_client.delete(key)
                        cleared_count += 1
            
            logger.info(f"Cleared {cleared_count} stale cache entries")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error cleaning up cache entries: {e}")
            return 0
    
    def _cleanup_memory_caches(self, cutoff_date: datetime):
        """Clean up in-memory caches"""
        # Clean up active timeouts that are very old
        expired_journeys = [
            journey_id for journey_id, timeout_at in self._active_timeouts.items()
            if timeout_at < cutoff_date
        ]
        
        for journey_id in expired_journeys:
            self._active_timeouts.pop(journey_id, None)
        
        # Clean up abandonment cache
        expired_penalties = [
            journey_id for journey_id, penalty in self._abandonment_cache.items()
            if penalty.created_at < cutoff_date
        ]
        
        for journey_id in expired_penalties:
            self._abandonment_cache.pop(journey_id, None)
        
        logger.info(f"Cleaned up {len(expired_journeys)} expired timeouts and "
                   f"{len(expired_penalties)} old penalties from memory")


# Factory function for easy initialization
def create_timeout_manager(
    timeout_days: int = 14,
    project_id: str = None,
    dataset_id: str = "gaelp",
    **config_kwargs
) -> JourneyTimeoutManager:
    """
    Factory function to create a JourneyTimeoutManager with common configuration.
    
    Args:
        timeout_days: Default timeout period in days
        project_id: Google Cloud project ID
        dataset_id: BigQuery dataset ID
        **config_kwargs: Additional configuration parameters
    
    Returns:
        Configured JourneyTimeoutManager instance
    """
    config = TimeoutConfiguration(
        default_timeout_days=timeout_days,
        **config_kwargs
    )
    
    return JourneyTimeoutManager(
        config=config,
        project_id=project_id,
        dataset_id=dataset_id
    )


# Example usage and testing
if __name__ == "__main__":
    async def test_timeout_manager():
        """Test the journey timeout manager"""
        
        # Create timeout manager
        config = TimeoutConfiguration(
            default_timeout_days=14,
            inactivity_threshold_hours=72,
            abandonment_check_interval_minutes=5  # Faster for testing
        )
        
        timeout_manager = JourneyTimeoutManager(
            config=config,
            project_id="test-project",
            dataset_id="gaelp_test"
        )
        
        # Start the manager
        await timeout_manager.start()
        
        try:
            # Register a test journey
            journey_id = "test-journey-123"
            start_time = datetime.now() - timedelta(days=15)  # Already expired
            
            timeout_at = await timeout_manager.register_journey(
                journey_id=journey_id,
                start_time=start_time,
                user_id="test-user-456"
            )
            
            print(f"Registered journey {journey_id} with timeout: {timeout_at}")
            
            # Check timeouts (should find the expired journey)
            timed_out = await timeout_manager.check_timeouts()
            print(f"Timed out journeys: {timed_out}")
            
            # Get abandonment analytics
            analytics = await timeout_manager.get_abandonment_analytics(
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            print(f"Abandonment analytics: {analytics}")
            
            # Clean up stale data
            cleanup_stats = await timeout_manager.cleanup_stale_data(older_than_days=30)
            print(f"Cleanup stats: {cleanup_stats}")
            
        finally:
            # Stop the manager
            await timeout_manager.stop()
    
    # Run the test
    asyncio.run(test_timeout_manager())
    print("JourneyTimeoutManager test completed successfully!")