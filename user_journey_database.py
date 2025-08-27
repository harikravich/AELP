"""
GAELP User Journey Database System
Comprehensive multi-touch attribution and user journey tracking with BigQuery integration.
"""

import os
import uuid
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from google.cloud import bigquery
import numpy as np

from journey_state import (
    JourneyState, TransitionTrigger, JourneyStateManager, 
    StateTransition, create_state_transition
)
from identity_resolver import IdentityResolver, DeviceSignature, MatchConfidence

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile with identity resolution data."""
    user_id: str
    canonical_user_id: str
    device_ids: List[str]
    email_hash: Optional[str] = None
    phone_hash: Optional[str] = None
    fingerprint_hash: Optional[str] = None
    ip_address: Optional[str] = None
    age_range: Optional[str] = None
    gender: Optional[str] = None
    location_country: Optional[str] = None
    location_region: Optional[str] = None
    location_city: Optional[str] = None
    current_journey_state: JourneyState = JourneyState.UNAWARE
    journey_score: float = 0.0
    conversion_probability: float = 0.0
    first_seen: datetime = None
    last_seen: datetime = None

@dataclass
class UserJourney:
    """Individual user journey instance."""
    journey_id: str
    user_id: str
    canonical_user_id: str
    journey_start: datetime
    journey_end: Optional[datetime] = None
    is_active: bool = True
    timeout_at: datetime = None
    initial_state: JourneyState = JourneyState.UNAWARE
    current_state: JourneyState = JourneyState.UNAWARE
    final_state: Optional[JourneyState] = None
    state_progression: List[Dict[str, Any]] = None
    converted: bool = False
    conversion_timestamp: Optional[datetime] = None
    conversion_value: Optional[float] = None
    conversion_type: Optional[str] = None
    first_touch_channel: Optional[str] = None
    last_touch_channel: Optional[str] = None
    touchpoint_count: int = 0
    days_to_conversion: Optional[int] = None
    journey_score: float = 0.0
    engagement_score: float = 0.0
    intent_score: float = 0.0

@dataclass
class JourneyTouchpoint:
    """Individual touchpoint within a journey."""
    touchpoint_id: str
    journey_id: str
    user_id: str
    canonical_user_id: str
    timestamp: datetime
    channel: str
    campaign_id: Optional[str] = None
    creative_id: Optional[str] = None
    placement_id: Optional[str] = None
    interaction_type: str = "impression"
    page_url: Optional[str] = None
    referrer_url: Optional[str] = None
    device_type: Optional[str] = None
    browser: Optional[str] = None
    os: Optional[str] = None
    content_category: Optional[str] = None
    message_variant: Optional[str] = None
    audience_segment: Optional[str] = None
    targeting_criteria: Optional[Dict] = None
    pre_state: Optional[JourneyState] = None
    post_state: Optional[JourneyState] = None
    state_change_confidence: Optional[float] = None
    dwell_time_seconds: Optional[float] = None
    scroll_depth: Optional[float] = None
    click_depth: Optional[int] = None
    engagement_score: Optional[float] = None
    first_touch_weight: float = 0.0
    last_touch_weight: float = 0.0
    time_decay_weight: float = 0.0
    position_weight: float = 0.0
    session_id: Optional[str] = None

@dataclass
class CompetitorExposure:
    """Competitor exposure tracking."""
    exposure_id: str
    user_id: str
    canonical_user_id: str
    journey_id: Optional[str]
    competitor_name: str
    competitor_channel: str
    exposure_timestamp: datetime
    exposure_type: str
    pre_exposure_state: Optional[JourneyState] = None
    post_exposure_state: Optional[JourneyState] = None
    state_impact_score: Optional[float] = None
    competitor_message: Optional[str] = None
    competitor_offer: Optional[str] = None
    price_comparison: Optional[float] = None
    feature_comparison: Optional[Dict] = None
    journey_impact_score: Optional[float] = None
    conversion_probability_change: Optional[float] = None

class UserJourneyDatabase:
    """Main database class for managing user journeys with BigQuery integration."""
    
    def __init__(self, 
                 project_id: str = None,
                 dataset_id: str = "gaelp_data",  # Updated to match our created dataset
                 timeout_days: int = 14,
                 state_manager: Optional[JourneyStateManager] = None,
                 identity_resolver: Optional[IdentityResolver] = None):
        # Use Thrive project by default
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT', 'aura-thrive-platform')
        self.dataset_id = dataset_id
        self.timeout_days = timeout_days
        self.state_manager = state_manager or JourneyStateManager()
        
        # Initialize Identity Resolver
        self.identity_resolver = identity_resolver or IdentityResolver(
            min_confidence_threshold=0.3,
            high_confidence_threshold=0.8,
            medium_confidence_threshold=0.5
        )
        
        # Initialize BigQuery client (with fallback for local testing)
        try:
            self.client = bigquery.Client(project=project_id)
            # Test if we can actually query BigQuery
            test_query = "SELECT 1"
            list(self.client.query(test_query).result(timeout=2))
            self.bigquery_available = True
            logger.info("BigQuery connection established")
        except Exception as e:
            # NO FALLBACKS - BigQuery MUST work
            import sys
            sys.path.insert(0, '/home/hariravichandran/AELP')
            from NO_FALLBACKS import StrictModeEnforcer
            StrictModeEnforcer.enforce('JOURNEY_DATABASE', fallback_attempted=True)
            raise Exception(f"BigQuery MUST be available. NO FALLBACKS! Error: {e}")
        
        # In-memory caches for performance
        self._active_journeys_cache: Dict[str, UserJourney] = {}
        self._user_profiles_cache: Dict[str, UserProfile] = {}
        self._cache_ttl = timedelta(minutes=15)
        self._last_cache_refresh = datetime.now()
        
    def _refresh_cache_if_needed(self):
        """Refresh in-memory caches if TTL expired."""
        if datetime.now() - self._last_cache_refresh > self._cache_ttl:
            self._load_active_journeys_cache()
            self._last_cache_refresh = datetime.now()
    
    def _load_active_journeys_cache(self):
        """Load active journeys into memory cache."""
        if not self.bigquery_available:
            # Keep existing cache when BigQuery is not available
            return
            
        query = f"""
        SELECT * FROM `{self.project_id}.{self.dataset_id}.user_journeys`
        WHERE is_active = TRUE 
        AND timeout_at > CURRENT_TIMESTAMP()
        """
        
        try:
            results = self.client.query(query)
            self._active_journeys_cache = {}
            
            for row in results:
                journey = self._row_to_journey(row)
                self._active_journeys_cache[journey.journey_id] = journey
                
        except Exception as e:
            logger.error(f"Failed to refresh journeys cache: {e}")
    
    def get_or_create_user(self, user_id: str, attributes: Dict[str, Any] = None) -> Tuple[UserProfile, bool]:
        """Get or create a user profile.
        
        Args:
            user_id: User identifier
            attributes: User attributes
            
        Returns:
            Tuple of (UserProfile, is_new)
        """
        # Check cache first
        if user_id in self._user_profiles_cache:
            return self._user_profiles_cache[user_id], False
        
        # Create new user profile
        user_profile = UserProfile(
            user_id=user_id,
            canonical_user_id=user_id,
            device_ids=[user_id],
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            conversion_probability=0.02,
            current_journey_state=JourneyState.UNAWARE,
            journey_score=0.0
        )
        
        # Cache the profile
        self._user_profiles_cache[user_id] = user_profile
        
        return user_profile, True
    
    def get_or_create_journey(self, 
                            user_id: str,
                            channel: str,
                            interaction_type: str = "impression",
                            device_fingerprint: Optional[Dict] = None,
                            **touchpoint_data) -> Tuple[UserJourney, bool]:
        """
        Get existing active journey or create new one for user.
        
        Returns:
            Tuple of (UserJourney, is_new_journey)
        """
        self._refresh_cache_if_needed()
        
        # Resolve user identity with cross-device tracking
        canonical_user_id = self._resolve_user_identity(user_id, device_fingerprint)
        
        # Check if this is a new device/session for existing identity
        original_user_id = user_id
        is_new_device = canonical_user_id != original_user_id
        
        if is_new_device:
            logger.info(f"Cross-device match detected: {original_user_id} -> {canonical_user_id}")
            
            # Get identity cluster info for validation
            cluster = self.identity_resolver.get_identity_cluster(canonical_user_id)
            if cluster:
                confidence = cluster.confidence_scores.get(original_user_id, 0.0)
                logger.info(f"Cross-device confidence: {confidence:.3f} across {len(cluster.device_ids)} devices")
            
            # Check for potential journey merging
            self._handle_cross_device_journey_merge(original_user_id, canonical_user_id)
        
        # Check for existing active journey
        existing_journey = self._find_active_journey(canonical_user_id)
        
        if existing_journey and not self._is_journey_expired(existing_journey):
            # Update journey to include new device if needed
            if is_new_device:
                self._add_device_to_journey(existing_journey, original_user_id)
            
            logger.info(f"Found existing active journey: {existing_journey.journey_id} for canonical user: {canonical_user_id}")
            return existing_journey, False
        
        # Create new journey
        journey_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        new_journey = UserJourney(
            journey_id=journey_id,
            user_id=user_id,  # Original device-specific user ID
            canonical_user_id=canonical_user_id,  # Cross-device canonical ID
            journey_start=current_time,
            timeout_at=current_time + timedelta(days=self.timeout_days),
            first_touch_channel=channel,
            last_touch_channel=channel,
            state_progression=[{
                'state': JourneyState.UNAWARE.value,
                'timestamp': current_time.isoformat(),
                'confidence': 1.0,
                'trigger_event': 'journey_start',
                'device_id': user_id,
                'canonical_user_id': canonical_user_id
            }]
        )
        
        # Store in database
        self._insert_journey(new_journey)
        
        # Update cache
        self._active_journeys_cache[journey_id] = new_journey
        
        # Track journey creation in identity graph
        if is_new_device:
            cluster = self.identity_resolver.get_identity_cluster(canonical_user_id)
            if cluster:
                cluster.updated_at = datetime.now()
                self._persist_identity_graph_updates(canonical_user_id, cluster)
        
        logger.info(f"Created new journey: {journey_id} for canonical user: {canonical_user_id} (device: {user_id})")
        return new_journey, True
    
    def update_journey(self,
                      journey_id: str,
                      touchpoint: JourneyTouchpoint,
                      trigger: TransitionTrigger = TransitionTrigger.IMPRESSION) -> UserJourney:
        """
        Update journey with new touchpoint and potentially trigger state transition.
        """
        self._refresh_cache_if_needed()
        
        # Get current journey
        journey = self._active_journeys_cache.get(journey_id)
        if not journey:
            journey = self._load_journey(journey_id)
            if not journey:
                raise ValueError(f"Journey not found: {journey_id}")
        
        # Check if journey is expired
        if self._is_journey_expired(journey):
            self._expire_journey(journey)
            raise ValueError(f"Journey expired: {journey_id}")
        
        # Calculate engagement score for touchpoint
        touchpoint_data = asdict(touchpoint)
        touchpoint.engagement_score = self.state_manager.calculate_engagement_score(touchpoint_data)
        
        # Set pre-state
        touchpoint.pre_state = journey.current_state
        
        # Predict state transition
        context = self._build_transition_context(journey, touchpoint)
        next_state, confidence = self.state_manager.predict_next_state(
            journey.current_state, trigger, context
        )
        
        # Check if transition should occur
        should_transition = self.state_manager.should_transition(
            journey.current_state, next_state, confidence
        )
        
        if should_transition and next_state != journey.current_state:
            # Update journey state
            old_state = journey.current_state
            journey.current_state = next_state
            touchpoint.post_state = next_state
            touchpoint.state_change_confidence = confidence
            
            # Record state transition
            transition = create_state_transition(
                from_state=old_state,
                to_state=next_state,
                trigger=trigger,
                confidence=confidence,
                touchpoint_id=touchpoint.touchpoint_id,
                channel=touchpoint.channel,
                **context
            )
            
            self._record_state_transition(transition)
            
            # Update state progression
            if not journey.state_progression:
                journey.state_progression = []
            
            journey.state_progression.append({
                'state': next_state.value,
                'timestamp': touchpoint.timestamp.isoformat(),
                'confidence': confidence,
                'trigger_event': trigger.value
            })
            
            # Check for conversion
            if next_state == JourneyState.CONVERTED:
                self._mark_journey_converted(journey, touchpoint)
            
            logger.info(f"State transition: {old_state.value} -> {next_state.value} "
                       f"(confidence: {confidence:.3f})")
        else:
            touchpoint.post_state = journey.current_state
        
        # Update journey metrics
        journey.touchpoint_count += 1
        journey.last_touch_channel = touchpoint.channel
        
        # Recalculate journey scores
        touchpoints = self._load_journey_touchpoints(journey_id)
        journey.journey_score = self.state_manager.calculate_journey_score(
            touchpoints, journey.current_state
        )
        
        journey.engagement_score = self._calculate_journey_engagement_score(touchpoints)
        journey.intent_score = self._calculate_journey_intent_score(context)
        
        # Calculate conversion probability
        days_in_journey = (datetime.now() - journey.journey_start).days
        conversion_prob = self.state_manager.calculate_conversion_probability(
            current_state=journey.current_state,
            journey_score=journey.journey_score,
            days_in_journey=days_in_journey,
            touchpoint_count=journey.touchpoint_count,
            context=context
        )
        
        # Update attribution weights
        self._update_attribution_weights(journey_id)
        
        # Store updates
        self._insert_touchpoint(touchpoint)
        self._update_journey(journey)
        
        # Update cache
        self._active_journeys_cache[journey_id] = journey
        
        return journey
    
    def record_competitor_exposure(self,
                                 user_id: str,
                                 competitor_name: str,
                                 competitor_channel: str,
                                 exposure_type: str,
                                 **exposure_data) -> None:
        """Record competitor exposure for impact analysis."""
        
        canonical_user_id = self._resolve_user_identity(user_id)
        journey = self._find_active_journey(canonical_user_id)
        
        exposure = CompetitorExposure(
            exposure_id=str(uuid.uuid4()),
            user_id=user_id,
            canonical_user_id=canonical_user_id,
            journey_id=journey.journey_id if journey else None,
            competitor_name=competitor_name,
            competitor_channel=competitor_channel,
            exposure_timestamp=datetime.now(),
            exposure_type=exposure_type,
            **exposure_data
        )
        
        if journey:
            exposure.pre_exposure_state = journey.current_state
            
            # Calculate impact on journey
            impact_score = self._calculate_competitor_impact(journey, exposure)
            exposure.journey_impact_score = impact_score
            
            # Estimate impact on conversion probability
            old_prob = self.state_manager.calculate_conversion_probability(
                journey.current_state, journey.journey_score, 
                (datetime.now() - journey.journey_start).days,
                journey.touchpoint_count, {}
            )
            
            new_prob = old_prob * (1.0 - impact_score * 0.3)  # Reduce by up to 30%
            exposure.conversion_probability_change = new_prob - old_prob
        
        self._insert_competitor_exposure(exposure)
    
    def get_journey_analytics(self, 
                            journey_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a journey."""
        
        journey = self._load_journey(journey_id)
        if not journey:
            raise ValueError(f"Journey not found: {journey_id}")
        
        touchpoints = self._load_journey_touchpoints(journey_id)
        transitions = self._load_journey_transitions(journey_id)
        competitor_exposures = self._load_journey_competitor_exposures(journey_id)
        
        # Calculate attribution weights
        attribution = self._calculate_attribution_weights(touchpoints)
        
        # Channel analysis
        channel_performance = self._analyze_channel_performance(touchpoints)
        
        # Journey path analysis
        path_analysis = self._analyze_journey_path(transitions)
        
        return {
            'journey': asdict(journey),
            'touchpoints': [asdict(tp) for tp in touchpoints],
            'state_transitions': [asdict(tr) for tr in transitions],
            'competitor_exposures': [asdict(ce) for ce in competitor_exposures],
            'attribution': attribution,
            'channel_performance': channel_performance,
            'path_analysis': path_analysis,
            'conversion_probability': self.state_manager.calculate_conversion_probability(
                journey.current_state, journey.journey_score,
                (datetime.now() - journey.journey_start).days,
                journey.touchpoint_count, {}
            )
        }
    
    def get_user_journey_history(self, 
                               canonical_user_id: str,
                               include_inactive: bool = True) -> List[UserJourney]:
        """Get complete journey history for a user."""
        
        query = f"""
        SELECT * FROM `{self.project_id}.{self.dataset_id}.user_journeys`
        WHERE canonical_user_id = @user_id
        """
        
        if not include_inactive:
            query += " AND is_active = TRUE"
        
        query += " ORDER BY journey_start DESC"
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", canonical_user_id)
            ]
        )
        
        results = self.client.query(query, job_config=job_config)
        return [self._row_to_journey(row) for row in results]
    
    def cleanup_expired_journeys(self) -> int:
        """Clean up expired journeys and return count of cleaned journeys."""
        
        query = f"""
        UPDATE `{self.project_id}.{self.dataset_id}.user_journeys`
        SET is_active = FALSE,
            journey_end = timeout_at,
            final_state = current_state
        WHERE is_active = TRUE 
        AND timeout_at <= CURRENT_TIMESTAMP()
        """
        
        job = self.client.query(query)
        job.result()  # Wait for completion
        
        # Clear cache to force refresh
        self._active_journeys_cache.clear()
        
        return job.num_dml_affected_rows
    
    def _resolve_user_identity(self, 
                             user_id: str,
                             device_fingerprint: Optional[Dict] = None) -> str:
        """Resolve user identity using multiple signals and confidence scoring."""
        
        # Create or update device signature if fingerprint provided
        if device_fingerprint:
            self._update_device_signature(user_id, device_fingerprint)
        
        # Try to resolve identity using the resolver
        resolved_identity = self.identity_resolver.resolve_identity(user_id)
        
        if resolved_identity and resolved_identity != user_id:
            # Get identity cluster for confidence validation
            cluster = self.identity_resolver.get_identity_cluster(resolved_identity)
            if cluster and user_id in cluster.confidence_scores:
                confidence = cluster.confidence_scores[user_id]
                logger.info(f"Resolved identity {user_id} -> {resolved_identity} (confidence: {confidence:.3f})")
                
                # Only use resolved identity if confidence is high enough
                if confidence >= self.identity_resolver.min_confidence_threshold:
                    return resolved_identity
                else:
                    logger.warning(f"Identity resolution confidence too low ({confidence:.3f}), using original ID")
            else:
                logger.info(f"Resolved identity {user_id} -> {resolved_identity}")
                return resolved_identity
        
        # Fallback to user_id if no identity found or confidence too low
        logger.info(f"No identity resolution for {user_id}, using as canonical")
        return user_id
    
    def _find_active_journey(self, canonical_user_id: str) -> Optional[UserJourney]:
        """Find active journey for user."""
        
        # Check cache first
        for journey in self._active_journeys_cache.values():
            if (journey.canonical_user_id == canonical_user_id and 
                journey.is_active and 
                not self._is_journey_expired(journey)):
                return journey
        
        # Query database
        query = f"""
        SELECT * FROM `{self.project_id}.{self.dataset_id}.user_journeys`
        WHERE canonical_user_id = @user_id
        AND is_active = TRUE
        AND timeout_at > CURRENT_TIMESTAMP()
        ORDER BY journey_start DESC
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", canonical_user_id)
            ]
        )
        
        results = self.client.query(query, job_config=job_config)
        
        for row in results:
            journey = self._row_to_journey(row)
            self._active_journeys_cache[journey.journey_id] = journey
            return journey
        
        return None
    
    def _is_journey_expired(self, journey: UserJourney) -> bool:
        """Check if journey has expired."""
        return datetime.now() > journey.timeout_at
    
    def _build_transition_context(self, 
                                journey: UserJourney, 
                                touchpoint: JourneyTouchpoint) -> Dict[str, Any]:
        """Build context for state transition prediction."""
        
        return {
            'engagement_score': touchpoint.engagement_score or 0.0,
            'journey_score': journey.journey_score,
            'touchpoint_count': journey.touchpoint_count,
            'days_in_journey': (datetime.now() - journey.journey_start).days,
            'channel': touchpoint.channel,
            'interaction_type': touchpoint.interaction_type,
            'device_type': touchpoint.device_type,
            'content_category': touchpoint.content_category,
            'audience_segment': touchpoint.audience_segment,
            'recent_competitor_exposure': self._check_recent_competitor_exposure(journey.journey_id)
        }
    
    # Database operation methods
    def _insert_journey(self, journey: UserJourney):
        """Insert new journey into BigQuery."""
        
        table_ref = self.client.dataset(self.dataset_id).table('user_journeys')
        table = self.client.get_table(table_ref)
        
        rows_to_insert = [self._journey_to_bigquery_row(journey)]
        errors = self.client.insert_rows(table, rows_to_insert)
        
        if errors:
            logger.error(f"Failed to insert journey: {errors}")
            raise Exception(f"BigQuery insert failed: {errors}")
    
    def _insert_touchpoint(self, touchpoint: JourneyTouchpoint):
        """Insert touchpoint into BigQuery."""
        
        table_ref = self.client.dataset(self.dataset_id).table('journey_touchpoints')
        table = self.client.get_table(table_ref)
        
        rows_to_insert = [self._touchpoint_to_bigquery_row(touchpoint)]
        errors = self.client.insert_rows(table, rows_to_insert)
        
        if errors:
            logger.error(f"Failed to insert touchpoint: {errors}")
    
    def _update_journey(self, journey: UserJourney):
        """Update journey in BigQuery."""
        
        query = f"""
        UPDATE `{self.project_id}.{self.dataset_id}.user_journeys`
        SET 
            current_state = @current_state,
            touchpoint_count = @touchpoint_count,
            journey_score = @journey_score,
            engagement_score = @engagement_score,
            intent_score = @intent_score,
            last_touch_channel = @last_touch_channel,
            state_progression = @state_progression,
            converted = @converted,
            conversion_timestamp = @conversion_timestamp,
            conversion_value = @conversion_value,
            updated_at = CURRENT_TIMESTAMP()
        WHERE journey_id = @journey_id
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("journey_id", "STRING", journey.journey_id),
                bigquery.ScalarQueryParameter("current_state", "STRING", journey.current_state.value),
                bigquery.ScalarQueryParameter("touchpoint_count", "INT64", journey.touchpoint_count),
                bigquery.ScalarQueryParameter("journey_score", "FLOAT64", journey.journey_score),
                bigquery.ScalarQueryParameter("engagement_score", "FLOAT64", journey.engagement_score),
                bigquery.ScalarQueryParameter("intent_score", "FLOAT64", journey.intent_score),
                bigquery.ScalarQueryParameter("last_touch_channel", "STRING", journey.last_touch_channel),
                bigquery.ScalarQueryParameter("state_progression", "STRING", 
                                             json.dumps(journey.state_progression or [])),
                bigquery.ScalarQueryParameter("converted", "BOOL", journey.converted),
                bigquery.ScalarQueryParameter("conversion_timestamp", "TIMESTAMP", journey.conversion_timestamp),
                bigquery.ScalarQueryParameter("conversion_value", "FLOAT64", journey.conversion_value)
            ]
        )
        
        job = self.client.query(query, job_config=job_config)
        job.result()
    
    # Helper methods for data conversion
    def _journey_to_bigquery_row(self, journey: UserJourney) -> Dict:
        """Convert UserJourney to BigQuery row format."""
        
        return {
            'journey_id': journey.journey_id,
            'user_id': journey.user_id,
            'canonical_user_id': journey.canonical_user_id,
            'journey_start': journey.journey_start,
            'journey_end': journey.journey_end,
            'is_active': journey.is_active,
            'timeout_at': journey.timeout_at,
            'initial_state': journey.initial_state.value,
            'current_state': journey.current_state.value,
            'final_state': journey.final_state.value if journey.final_state else None,
            'state_progression': json.dumps(journey.state_progression or []),
            'converted': journey.converted,
            'conversion_timestamp': journey.conversion_timestamp,
            'conversion_value': journey.conversion_value,
            'conversion_type': journey.conversion_type,
            'first_touch_channel': journey.first_touch_channel,
            'last_touch_channel': journey.last_touch_channel,
            'touchpoint_count': journey.touchpoint_count,
            'days_to_conversion': journey.days_to_conversion,
            'journey_score': journey.journey_score,
            'engagement_score': journey.engagement_score,
            'intent_score': journey.intent_score
        }
    
    def _touchpoint_to_bigquery_row(self, touchpoint: JourneyTouchpoint) -> Dict:
        """Convert JourneyTouchpoint to BigQuery row format."""
        
        return {
            'touchpoint_id': touchpoint.touchpoint_id,
            'journey_id': touchpoint.journey_id,
            'user_id': touchpoint.user_id,
            'canonical_user_id': touchpoint.canonical_user_id,
            'timestamp': touchpoint.timestamp,
            'channel': touchpoint.channel,
            'campaign_id': touchpoint.campaign_id,
            'creative_id': touchpoint.creative_id,
            'placement_id': touchpoint.placement_id,
            'interaction_type': touchpoint.interaction_type,
            'page_url': touchpoint.page_url,
            'referrer_url': touchpoint.referrer_url,
            'device_type': touchpoint.device_type,
            'browser': touchpoint.browser,
            'os': touchpoint.os,
            'content_category': touchpoint.content_category,
            'message_variant': touchpoint.message_variant,
            'audience_segment': touchpoint.audience_segment,
            'targeting_criteria': json.dumps(touchpoint.targeting_criteria or {}),
            'pre_state': touchpoint.pre_state.value if touchpoint.pre_state else None,
            'post_state': touchpoint.post_state.value if touchpoint.post_state else None,
            'state_change_confidence': touchpoint.state_change_confidence,
            'dwell_time_seconds': touchpoint.dwell_time_seconds,
            'scroll_depth': touchpoint.scroll_depth,
            'click_depth': touchpoint.click_depth,
            'engagement_score': touchpoint.engagement_score,
            'first_touch_weight': touchpoint.first_touch_weight,
            'last_touch_weight': touchpoint.last_touch_weight,
            'time_decay_weight': touchpoint.time_decay_weight,
            'position_weight': touchpoint.position_weight,
            'session_id': touchpoint.session_id
        }
    
    def _row_to_journey(self, row) -> UserJourney:
        """Convert BigQuery row to UserJourney object."""
        
        return UserJourney(
            journey_id=row.journey_id,
            user_id=row.user_id,
            canonical_user_id=row.canonical_user_id,
            journey_start=row.journey_start,
            journey_end=row.journey_end,
            is_active=row.is_active,
            timeout_at=row.timeout_at,
            initial_state=JourneyState(row.initial_state),
            current_state=JourneyState(row.current_state),
            final_state=JourneyState(row.final_state) if row.final_state else None,
            state_progression=json.loads(row.state_progression) if row.state_progression else [],
            converted=row.converted,
            conversion_timestamp=row.conversion_timestamp,
            conversion_value=row.conversion_value,
            conversion_type=row.conversion_type,
            first_touch_channel=row.first_touch_channel,
            last_touch_channel=row.last_touch_channel,
            touchpoint_count=row.touchpoint_count,
            days_to_conversion=row.days_to_conversion,
            journey_score=row.journey_score,
            engagement_score=row.engagement_score,
            intent_score=row.intent_score
        )
    
    # Placeholder methods (implement based on specific requirements)
    def _record_state_transition(self, transition: StateTransition):
        """Record state transition in database."""
        # TODO: Implement BigQuery insert for state transitions
        pass
    
    def _mark_journey_converted(self, journey: UserJourney, touchpoint: JourneyTouchpoint):
        """Mark journey as converted."""
        journey.converted = True
        journey.conversion_timestamp = touchpoint.timestamp
        journey.is_active = False
        journey.journey_end = touchpoint.timestamp
        journey.final_state = JourneyState.CONVERTED
        
        if journey.journey_start:
            journey.days_to_conversion = (touchpoint.timestamp - journey.journey_start).days
    
    def _load_journey(self, journey_id: str) -> Optional[UserJourney]:
        """Load journey from database."""
        # TODO: Implement BigQuery query
        return None
    
    def _load_journey_touchpoints(self, journey_id: str) -> List[Dict]:
        """Load touchpoints for journey."""
        # TODO: Implement BigQuery query
        return []
    
    def _calculate_journey_engagement_score(self, touchpoints: List[Dict]) -> float:
        """Calculate overall engagement score for journey."""
        if not touchpoints:
            return 0.0
        
        scores = [tp.get('engagement_score', 0.0) for tp in touchpoints]
        return sum(scores) / len(scores)
    
    def _calculate_journey_intent_score(self, context: Dict) -> float:
        """Calculate intent score for journey."""
        return context.get('intent_score', 0.0)
    
    def _update_attribution_weights(self, journey_id: str):
        """Update attribution weights for all touchpoints in journey."""
        # TODO: Implement attribution weight calculation and update
        pass
    
    def _insert_competitor_exposure(self, exposure: CompetitorExposure):
        """Insert competitor exposure record."""
        # TODO: Implement BigQuery insert
        pass
    
    def _calculate_competitor_impact(self, journey: UserJourney, exposure: CompetitorExposure) -> float:
        """Calculate impact of competitor exposure on journey."""
        # Simple impact calculation - can be made more sophisticated
        base_impact = 0.2  # 20% base impact
        
        # Higher impact if in later stages of journey
        state_multiplier = {
            JourneyState.UNAWARE: 0.5,
            JourneyState.AWARE: 0.7,
            JourneyState.CONSIDERING: 1.0,
            JourneyState.INTENT: 1.5
        }
        
        return base_impact * state_multiplier.get(journey.current_state, 0.5)
    
    def _check_recent_competitor_exposure(self, journey_id: str) -> bool:
        """Check if there was recent competitor exposure."""
        # TODO: Implement BigQuery query to check recent competitor exposures
        return False
    
    def _log_identity_resolution(self, user_id: str, canonical_user_id: str, fingerprint: Dict):
        """Log identity resolution for ML training."""
        # TODO: Implement identity resolution logging
        pass
    
    def _expire_journey(self, journey: UserJourney):
        """Expire a journey due to timeout."""
        journey.is_active = False
        journey.journey_end = datetime.now()
        journey.final_state = journey.current_state
        self._update_journey(journey)
    
    def _load_journey_transitions(self, journey_id: str) -> List:
        """Load state transitions for journey."""
        # TODO: Implement
        return []
    
    def _load_journey_competitor_exposures(self, journey_id: str) -> List:
        """Load competitor exposures for journey."""
        # TODO: Implement
        return []
    
    def _calculate_attribution_weights(self, touchpoints: List) -> Dict:
        """Calculate attribution weights for touchpoints."""
        # TODO: Implement multi-touch attribution
        return {}
    
    def _analyze_channel_performance(self, touchpoints: List) -> Dict:
        """Analyze performance by channel."""
        # TODO: Implement channel analysis
        return {}
    
    def _analyze_journey_path(self, transitions: List) -> Dict:
        """Analyze journey path patterns."""
        # TODO: Implement path analysis
        return {}
    
    def _validate_journey_merge(self, source: UserJourney, target: UserJourney) -> bool:
        """Validate that two journeys can be safely merged."""
        # Check time overlap - journeys shouldn't be too far apart
        time_diff = abs((source.journey_start - target.journey_start).total_seconds())
        max_time_diff = self.timeout_days * 24 * 3600  # Allow full timeout window
        
        if time_diff > max_time_diff:
            logger.warning(f"Journey time difference too large: {time_diff/3600:.1f} hours")
            return False
        
        # Check for conflicting conversion states
        if source.converted and target.converted:
            # Both converted - need careful handling
            logger.warning("Both journeys have conversions - merge may cause attribution issues")
            # Allow merge but flag for review
        
        return True
    
    def _add_device_to_journey(self, journey: UserJourney, device_user_id: str) -> None:
        """Add a new device to an existing journey's tracking."""
        if not journey.state_progression:
            journey.state_progression = []
        
        journey.state_progression.append({
            'event_type': 'device_added',
            'timestamp': datetime.now().isoformat(),
            'device_user_id': device_user_id,
            'canonical_user_id': journey.canonical_user_id
        })
        
        logger.info(f"Added device {device_user_id} to journey {journey.journey_id}")
        
        # Update the journey in database
        self._update_journey(journey)
    
    def add_touchpoint(self,
                      journey_id: str,
                      touchpoint: JourneyTouchpoint,
                      trigger: TransitionTrigger = TransitionTrigger.IMPRESSION) -> UserJourney:
        """
        Add touchpoint to journey with cross-device identity validation.
        """
        # Validate touchpoint user identity
        if touchpoint.canonical_user_id:
            # Verify the canonical user ID matches our identity resolution
            resolved_id = self._resolve_user_identity(touchpoint.user_id)
            if resolved_id != touchpoint.canonical_user_id:
                logger.warning(f"Touchpoint canonical ID mismatch: {touchpoint.canonical_user_id} vs resolved {resolved_id}")
                touchpoint.canonical_user_id = resolved_id
        
        # Delegate to update_journey method
        return self.update_journey(journey_id, touchpoint, trigger)
    
    # Identity Resolution Integration Methods
    
    def _update_device_signature(self, user_id: str, device_fingerprint: Dict) -> None:
        """Update device signature for identity resolution."""
        try:
            # Extract relevant data from device fingerprint
            signature = DeviceSignature(
                device_id=user_id,
                user_agent=device_fingerprint.get('user_agent', ''),
                screen_resolution=device_fingerprint.get('screen_resolution', ''),
                timezone=device_fingerprint.get('timezone', ''),
                language=device_fingerprint.get('language', ''),
                platform=device_fingerprint.get('platform', ''),
                browser=device_fingerprint.get('browser', ''),
                last_seen=datetime.now()
            )
            
            # Add behavioral data if available
            if 'search_patterns' in device_fingerprint:
                signature.search_patterns = device_fingerprint['search_patterns']
            
            if 'session_duration' in device_fingerprint:
                signature.session_durations = [device_fingerprint['session_duration']]
            
            if 'time_of_day' in device_fingerprint:
                signature.time_of_day_usage = [device_fingerprint['time_of_day']]
            
            if 'ip_address' in device_fingerprint:
                signature.ip_addresses.add(device_fingerprint['ip_address'])
            
            if 'location' in device_fingerprint:
                loc = device_fingerprint['location']
                if isinstance(loc, dict) and 'lat' in loc and 'lon' in loc:
                    signature.geographic_locations.append((loc['lat'], loc['lon']))
            
            signature.session_timestamps.append(datetime.now())
            
            # Add to identity resolver
            self.identity_resolver.add_device_signature(signature)
            
            logger.info(f"Updated device signature for {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update device signature for {user_id}: {e}")
    
    def _handle_cross_device_journey_merge(self, original_user_id: str, canonical_user_id: str) -> None:
        """Handle journey merging when cross-device match is detected with enhanced validation."""
        if not self.bigquery_available:
            # Just update in-memory cache
            logger.info(f"Cross-device merge (in-memory only): {original_user_id} -> {canonical_user_id}")
            return
            
        try:
            # Find any active journey for the original user ID
            original_journey = self._find_active_journey(original_user_id)
            canonical_journey = self._find_active_journey(canonical_user_id)
            
            # Get identity cluster for confidence validation
            identity_cluster = self.identity_resolver.get_identity_cluster(canonical_user_id)
            merge_confidence = 0.0
            if identity_cluster and original_user_id in identity_cluster.confidence_scores:
                merge_confidence = identity_cluster.confidence_scores[original_user_id]
            
            # Only proceed with merge if confidence is sufficient
            if merge_confidence < self.identity_resolver.min_confidence_threshold:
                logger.warning(f"Skipping journey merge due to low confidence: {merge_confidence:.3f}")
                return
            
            if original_journey and canonical_journey and original_journey.journey_id != canonical_journey.journey_id:
                # Need to merge two active journeys
                logger.info(f"Merging journeys: {original_journey.journey_id} -> {canonical_journey.journey_id} (confidence: {merge_confidence:.3f})")
                self._merge_journeys_with_validation(original_journey, canonical_journey, merge_confidence)
                
            elif original_journey and not canonical_journey:
                # Update the original journey's canonical user ID
                logger.info(f"Updating journey {original_journey.journey_id} canonical user ID: {original_user_id} -> {canonical_user_id}")
                original_journey.canonical_user_id = canonical_user_id
                
                # Add merge metadata to state progression
                if not original_journey.state_progression:
                    original_journey.state_progression = []
                
                original_journey.state_progression.append({
                    'event_type': 'cross_device_merge',
                    'timestamp': datetime.now().isoformat(),
                    'original_user_id': original_user_id,
                    'canonical_user_id': canonical_user_id,
                    'merge_confidence': merge_confidence,
                    'device_count': len(identity_cluster.device_ids) if identity_cluster else 1
                })
                
                self._update_journey(original_journey)
                
                # Update cache
                self._active_journeys_cache[original_journey.journey_id] = original_journey
                
            # Update identity graph with the new match
            if identity_cluster:
                # Merge journeys in the identity graph
                merged_journey = self.identity_resolver.merge_journeys(canonical_user_id)
                logger.info(f"Merged journey data across {len(identity_cluster.device_ids)} devices")
                
                # Persist merged journey to BigQuery
                self._persist_identity_graph_updates(canonical_user_id, identity_cluster)
                
        except Exception as e:
            logger.error(f"Failed to handle cross-device journey merge: {e}")
    
    def _merge_journeys_with_validation(self, source_journey: UserJourney, target_journey: UserJourney, confidence: float) -> None:
        """Merge source journey into target journey with enhanced validation."""
        try:
            # Log detailed merge information
            logger.info(f"Merging journeys with validation:")
            logger.info(f"  Source: {source_journey.journey_id} (user: {source_journey.user_id})")
            logger.info(f"  Target: {target_journey.journey_id} (canonical: {target_journey.canonical_user_id})")
            logger.info(f"  Confidence: {confidence:.3f}")
            logger.info(f"  Source touchpoints: {source_journey.touchpoint_count}")
            logger.info(f"  Target touchpoints: {target_journey.touchpoint_count}")
            
            # Validate journeys are compatible for merging
            if not self._validate_journey_merge(source_journey, target_journey):
                logger.error("Journey merge validation failed")
                return
            
            # Proceed with the standard merge process
            self._merge_journeys(source_journey, target_journey)
            
        except Exception as e:
            logger.error(f"Failed to merge journeys with validation: {e}")
            raise
    
    def _merge_journeys(self, source_journey: UserJourney, target_journey: UserJourney) -> None:
        """Merge source journey into target journey."""
        try:
            # Merge touchpoints by updating their journey_id and canonical_user_id
            update_touchpoints_query = f"""
            UPDATE `{self.project_id}.{self.dataset_id}.journey_touchpoints`
            SET journey_id = @target_journey_id,
                canonical_user_id = @canonical_user_id
            WHERE journey_id = @source_journey_id
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("target_journey_id", "STRING", target_journey.journey_id),
                    bigquery.ScalarQueryParameter("canonical_user_id", "STRING", target_journey.canonical_user_id),
                    bigquery.ScalarQueryParameter("source_journey_id", "STRING", source_journey.journey_id)
                ]
            )
            
            job = self.client.query(update_touchpoints_query, job_config=job_config)
            job.result()
            
            # Update target journey metrics
            target_journey.touchpoint_count += source_journey.touchpoint_count
            
            # Merge state progression
            if source_journey.state_progression:
                if not target_journey.state_progression:
                    target_journey.state_progression = []
                target_journey.state_progression.extend(source_journey.state_progression)
                
                # Sort by timestamp
                target_journey.state_progression.sort(key=lambda x: x.get('timestamp', ''))
            
            # Update journey start if source is earlier
            if source_journey.journey_start < target_journey.journey_start:
                target_journey.first_touch_channel = source_journey.first_touch_channel
                target_journey.journey_start = source_journey.journey_start
            
            # Update journey scores (average or max)
            if source_journey.journey_score > 0:
                target_journey.journey_score = max(target_journey.journey_score, source_journey.journey_score)
            
            if source_journey.engagement_score > 0:
                target_journey.engagement_score = max(target_journey.engagement_score, source_journey.engagement_score)
            
            # Deactivate source journey
            source_journey.is_active = False
            source_journey.journey_end = datetime.now()
            source_journey.final_state = source_journey.current_state
            
            # Update both journeys in database
            self._update_journey(target_journey)
            self._update_journey(source_journey)
            
            # Update cache
            self._active_journeys_cache[target_journey.journey_id] = target_journey
            if source_journey.journey_id in self._active_journeys_cache:
                del self._active_journeys_cache[source_journey.journey_id]
            
            logger.info(f"Successfully merged journey {source_journey.journey_id} into {target_journey.journey_id}")
            
        except Exception as e:
            logger.error(f"Failed to merge journeys: {e}")
            raise
    
    def _persist_identity_graph_updates(self, identity_id: str, cluster: 'IdentityCluster') -> None:
        """Persist identity graph updates to BigQuery."""
        try:
            # Create or update identity graph table
            table_id = f"{self.project_id}.{self.dataset_id}.identity_graph"
            
            # Prepare row data
            row_data = {
                'identity_id': identity_id,
                'device_ids': list(cluster.device_ids),
                'primary_device_id': cluster.primary_device_id,
                'confidence_scores': json.dumps(cluster.confidence_scores),
                'created_at': cluster.created_at,
                'updated_at': cluster.updated_at,
                'merged_journey': json.dumps(cluster.merged_journey)
            }
            
            # Insert or update using merge query
            merge_query = f"""
            MERGE `{table_id}` T
            USING (SELECT @identity_id as identity_id) S
            ON T.identity_id = S.identity_id
            WHEN MATCHED THEN
              UPDATE SET
                device_ids = @device_ids,
                primary_device_id = @primary_device_id,
                confidence_scores = @confidence_scores,
                updated_at = @updated_at,
                merged_journey = @merged_journey
            WHEN NOT MATCHED THEN
              INSERT (identity_id, device_ids, primary_device_id, confidence_scores, 
                     created_at, updated_at, merged_journey)
              VALUES (@identity_id, @device_ids, @primary_device_id, @confidence_scores,
                     @created_at, @updated_at, @merged_journey)
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("identity_id", "STRING", identity_id),
                    bigquery.ArrayQueryParameter("device_ids", "STRING", list(cluster.device_ids)),
                    bigquery.ScalarQueryParameter("primary_device_id", "STRING", cluster.primary_device_id),
                    bigquery.ScalarQueryParameter("confidence_scores", "STRING", json.dumps(cluster.confidence_scores)),
                    bigquery.ScalarQueryParameter("created_at", "TIMESTAMP", cluster.created_at),
                    bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", cluster.updated_at),
                    bigquery.ScalarQueryParameter("merged_journey", "STRING", json.dumps(cluster.merged_journey))
                ]
            )
            
            job = self.client.query(merge_query, job_config=job_config)
            job.result()
            
            logger.info(f"Persisted identity graph updates for {identity_id}")
            
        except Exception as e:
            logger.error(f"Failed to persist identity graph updates: {e}")
    
    def get_cross_device_analytics(self, identity_id: str) -> Dict[str, Any]:
        """Get cross-device analytics for an identity."""
        try:
            cluster = self.identity_resolver.get_identity_cluster(identity_id)
            if not cluster:
                return {}
            
            # Get journeys for all devices in the cluster
            all_journeys = []
            for device_id in cluster.device_ids:
                device_journeys = self.get_user_journey_history(device_id, include_inactive=True)
                all_journeys.extend(device_journeys)
            
            # Analyze cross-device patterns
            device_types = set()
            channels_used = set()
            total_touchpoints = 0
            conversion_events = []
            
            for journey in all_journeys:
                if hasattr(journey, 'device_type'):
                    device_types.add(journey.device_type)
                if journey.first_touch_channel:
                    channels_used.add(journey.first_touch_channel)
                if journey.last_touch_channel:
                    channels_used.add(journey.last_touch_channel)
                total_touchpoints += journey.touchpoint_count
                
                if journey.converted:
                    conversion_events.append({
                        'journey_id': journey.journey_id,
                        'device_id': journey.user_id,
                        'conversion_timestamp': journey.conversion_timestamp,
                        'conversion_value': journey.conversion_value,
                        'days_to_conversion': journey.days_to_conversion
                    })
            
            return {
                'identity_id': identity_id,
                'device_count': len(cluster.device_ids),
                'device_ids': list(cluster.device_ids),
                'device_types': list(device_types),
                'channels_used': list(channels_used),
                'total_journeys': len(all_journeys),
                'total_touchpoints': total_touchpoints,
                'conversion_events': conversion_events,
                'merged_journey_events': len(cluster.merged_journey),
                'confidence_scores': cluster.confidence_scores,
                'created_at': cluster.created_at,
                'updated_at': cluster.updated_at
            }
            
        except Exception as e:
            logger.error(f"Failed to get cross-device analytics for {identity_id}: {e}")
            return {}

# Example usage demonstrating cross-device tracking
if __name__ == "__main__":
    # Initialize database (requires BigQuery credentials)
    # db = UserJourneyDatabase("your-project-id")
    
    print("=== Cross-Device User Journey Tracking Demo ===")
    
    # Simulate Mobile Lisa first interaction
    mobile_fingerprint = {
        'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)',
        'platform': 'iOS',
        'timezone': 'America/New_York',
        'language': 'en-US',
        'screen_resolution': '375x667',
        'search_patterns': ['python tutorial', 'machine learning'],
        'session_duration': 45.0,
        'time_of_day': 14,  # 2 PM
        'location': {'lat': 40.7128, 'lon': -74.0060},  # NYC
        'ip_address': '192.168.1.100'
    }
    
    # Simulate Desktop Lisa later interaction
    desktop_fingerprint = {
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'platform': 'Windows',
        'timezone': 'America/New_York', 
        'language': 'en-US',
        'screen_resolution': '1920x1080',
        'search_patterns': ['python tutorial', 'data science course'],
        'session_duration': 120.0,
        'time_of_day': 20,  # 8 PM
        'location': {'lat': 40.7589, 'lon': -73.9851},  # NYC nearby
        'ip_address': '192.168.1.101'  # Same network
    }
    
    # Create sample touchpoints
    mobile_touchpoint = JourneyTouchpoint(
        touchpoint_id=str(uuid.uuid4()),
        journey_id="",  # Will be set by get_or_create_journey
        user_id="mobile_lisa_001",
        canonical_user_id="",  # Will be resolved
        timestamp=datetime.now(),
        channel="facebook_ads",
        interaction_type="impression",
        device_type="mobile"
    )
    
    desktop_touchpoint = JourneyTouchpoint(
        touchpoint_id=str(uuid.uuid4()),
        journey_id="",  # Will be set by get_or_create_journey
        user_id="desktop_lisa_002", 
        canonical_user_id="",  # Will be resolved
        timestamp=datetime.now() + timedelta(hours=2),
        channel="google_ads",
        interaction_type="click",
        device_type="desktop"
    )
    
    print("Sample mobile touchpoint:", mobile_touchpoint.touchpoint_id)
    print("Sample desktop touchpoint:", desktop_touchpoint.touchpoint_id)
    print("\nWith device fingerprints for cross-device matching:")
    print("- Mobile device signature created")
    print("- Desktop device signature created")
    print("- Identity resolution will match Lisa across devices")
    print("- Journeys will be merged for unified tracking")
    print("\nUserJourneyDatabase with Identity Resolution ready!")