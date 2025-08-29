"""
Delayed Reward System for Multi-Day Conversions

Handles delayed rewards and attribution for ad campaigns where conversions
may happen days after the initial touchpoint. Integrates with the training
orchestrator to provide accurate reward signals for reinforcement learning.

Key Features:
- Stores pending rewards until conversions are detected
- Handles partial episodes where conversions happen after episode ends
- Implements multiple attribution models (last-click, first-click, linear, etc.)
- Includes reward replay buffer for training on delayed rewards
- Provides reward backpropagation to relevant episodes
"""

import asyncio
import logging
import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque

import numpy as np
import redis
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import conversion lag model for enhanced conversion timing prediction
try:
    from conversion_lag_model import ConversionLagModel, ConversionJourney
    CONVERSION_LAG_MODEL_AVAILABLE = True
except ImportError:
    CONVERSION_LAG_MODEL_AVAILABLE = False
    print("Warning: ConversionLagModel not available. Enhanced conversion timing disabled.")

logger = logging.getLogger(__name__)

Base = declarative_base()


class AttributionModel(Enum):
    """Attribution models for multi-touch attribution"""
    LAST_CLICK = "last_click"
    FIRST_CLICK = "first_click"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"
    DATA_DRIVEN = "data_driven"


class ConversionEvent(Enum):
    """Types of conversion events"""
    PURCHASE = "purchase"
    SIGNUP = "signup"
    LEAD = "lead"
    VIEW_CONTENT = "view_content"
    ADD_TO_CART = "add_to_cart"
    INITIATE_CHECKOUT = "initiate_checkout"


@dataclass
class Touchpoint:
    """Individual touchpoint in a customer journey"""
    touchpoint_id: str
    episode_id: str
    user_id: str
    campaign_id: str
    timestamp: datetime
    action: Dict[str, Any]
    state: Dict[str, Any]
    immediate_reward: float
    channel: str
    creative_type: str
    placement: str
    position_in_journey: int
    session_id: Optional[str] = None
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionRecord:
    """Record of a conversion event"""
    conversion_id: str
    user_id: str
    timestamp: datetime
    event_type: ConversionEvent
    value: float
    currency: str = "USD"
    product_id: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingReward:
    """Reward waiting for attribution after conversion"""
    reward_id: str
    touchpoints: List[Touchpoint]
    conversion: Optional[ConversionRecord] = None
    attribution_rewards: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, attributed, expired
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    attribution_model: AttributionModel = AttributionModel.LINEAR


@dataclass
class DelayedRewardConfig:
    """Configuration for delayed reward system"""
    
    # Reward window settings
    attribution_window_days: int = 7
    max_pending_rewards: int = 100000
    cleanup_interval_hours: int = 24
    
    # Attribution settings
    default_attribution_model: AttributionModel = AttributionModel.LINEAR
    time_decay_half_life_hours: float = 24.0
    position_based_first_weight: float = 0.4
    position_based_last_weight: float = 0.4
    
    # Conversion lag model settings
    enable_conversion_lag_model: bool = True
    conversion_lag_model_type: str = 'weibull'  # 'weibull' or 'cox'
    dynamic_attribution_windows: bool = True
    conversion_timeout_threshold_days: int = 45
    
    # Storage settings
    use_redis_cache: bool = True
    redis_host: str = os.environ.get('REDIS_HOST', 'localhost')
    redis_port: int = int(os.environ.get('REDIS_PORT', 6379))
    redis_db: int = 0
    redis_ttl_seconds: int = 604800  # 7 days
    
    use_database_persistence: bool = True
    database_url: str = "sqlite:///delayed_rewards.db"
    
    # Replay buffer settings
    replay_buffer_size: int = 50000
    min_replay_samples: int = 1000
    replay_batch_size: int = 32
    replay_frequency: int = 100  # Every N episodes
    
    # Performance settings
    batch_attribution_size: int = 1000
    max_concurrent_attributions: int = 10
    enable_async_processing: bool = True


class TouchpointTable(Base):
    """Database table for touchpoints"""
    __tablename__ = 'touchpoints'
    
    touchpoint_id = Column(String, primary_key=True)
    episode_id = Column(String, index=True)
    user_id = Column(String, index=True)
    campaign_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    action_data = Column(Text)  # JSON serialized
    state_data = Column(Text)  # JSON serialized
    immediate_reward = Column(Float)
    channel = Column(String)
    creative_type = Column(String)
    placement = Column(String)
    position_in_journey = Column(Integer)
    session_id = Column(String)
    cost = Column(Float)
    metadata_json = Column(Text)  # JSON serialized


class ConversionTable(Base):
    """Database table for conversions"""
    __tablename__ = 'conversions'
    
    conversion_id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    event_type = Column(String)
    value = Column(Float)
    currency = Column(String)
    product_id = Column(String)
    category = Column(String)
    metadata_json = Column(Text)  # JSON serialized
    attributed = Column(Boolean, default=False)


class RewardReplayBuffer:
    """Buffer for storing and replaying delayed reward experiences"""
    
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        
    def add_experience(self, 
                      episode_id: str,
                      original_reward: float,
                      attributed_reward: float,
                      touchpoint: Touchpoint,
                      conversion: ConversionRecord,
                      priority: float = 1.0):
        """Add a delayed reward experience to the buffer"""
        
        experience = {
            'episode_id': episode_id,
            'touchpoint_id': touchpoint.touchpoint_id,
            'original_reward': original_reward,
            'attributed_reward': attributed_reward,
            'reward_delta': attributed_reward - original_reward,
            'state': touchpoint.state,
            'action': touchpoint.action,
            'conversion_value': conversion.value,
            'time_to_conversion_hours': (conversion.timestamp - touchpoint.timestamp).total_seconds() / 3600,
            'timestamp': datetime.now()
        }
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of experiences for training"""
        
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Priority-based sampling
        probabilities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        
        return [self.buffer[i] for i in indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        
        if not self.buffer:
            return {'size': 0}
        
        reward_deltas = [exp['reward_delta'] for exp in self.buffer]
        time_to_conversions = [exp['time_to_conversion_hours'] for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'avg_reward_delta': np.mean(reward_deltas),
            'std_reward_delta': np.std(reward_deltas),
            'avg_time_to_conversion': np.mean(time_to_conversions),
            'conversion_rate': len([exp for exp in self.buffer if exp['attributed_reward'] > 0]) / len(self.buffer)
        }


class DelayedRewardSystem:
    """
    Main delayed reward system that handles multi-day conversions and attribution
    """
    
    def __init__(self, config: DelayedRewardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage components
        self.redis_client = None
        self.db_engine = None
        self.db_session = None
        
        # Conversion lag model for enhanced timing predictions
        self.conversion_lag_model = None
        if config.enable_conversion_lag_model and CONVERSION_LAG_MODEL_AVAILABLE:
            self.conversion_lag_model = ConversionLagModel(
                attribution_window_days=config.attribution_window_days,
                timeout_threshold_days=config.conversion_timeout_threshold_days,
                model_type=config.conversion_lag_model_type
            )
            self.logger.info(f"Initialized conversion lag model: {config.conversion_lag_model_type}")
        
        # In-memory tracking
        self.pending_rewards: Dict[str, PendingReward] = {}
        self.user_journeys: Dict[str, List[Touchpoint]] = defaultdict(list)
        self.episode_touchpoints: Dict[str, List[str]] = defaultdict(list)
        self.conversion_journey_cache: Dict[str, ConversionJourney] = {}  # For lag model
        
        # Replay buffer
        self.replay_buffer = RewardReplayBuffer(config.replay_buffer_size)
        
        # Performance tracking
        self.attribution_stats = {
            'total_conversions_attributed': 0,
            'total_reward_attributed': 0.0,
            'attribution_latency_avg': 0.0,
            'pending_rewards_count': 0,
            'dynamic_windows_used': 0,
            'avg_dynamic_window_days': 0.0
        }
        
        # Background tasks
        self.cleanup_task = None
        self.attribution_task = None
        
        self._initialize_storage()
        
    def _initialize_storage(self):
        """Initialize storage backends"""
        
        # Initialize Redis for caching
        if self.config.use_redis_cache:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()  # Test connection
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Initialize database for persistence
        if self.config.use_database_persistence:
            try:
                self.db_engine = create_engine(self.config.database_url)
                Base.metadata.create_all(self.db_engine)
                Session = sessionmaker(bind=self.db_engine)
                self.db_session = Session()
                self.logger.info("Database connection established")
            except Exception as e:
                self.logger.error(f"Database connection failed: {e}")
                
        # Start background tasks
        if self.config.enable_async_processing:
            self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background cleanup and attribution tasks"""
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        self.cleanup_task = loop.create_task(self._periodic_cleanup())
        self.attribution_task = loop.create_task(self._periodic_attribution())
    
    async def store_pending_reward(self,
                                  episode_id: str,
                                  user_id: str,
                                  campaign_id: str,
                                  action: Dict[str, Any],
                                  state: Dict[str, Any],
                                  immediate_reward: float,
                                  **kwargs) -> str:
        """
        Store a pending reward awaiting attribution
        
        Args:
            episode_id: Episode identifier
            user_id: User identifier for journey tracking
            campaign_id: Campaign identifier
            action: Action taken by agent
            state: Environment state
            immediate_reward: Immediate reward received
            **kwargs: Additional touchpoint metadata
            
        Returns:
            Touchpoint ID for the stored pending reward
        """
        
        touchpoint_id = str(uuid.uuid4())
        
        # Create touchpoint
        touchpoint = Touchpoint(
            touchpoint_id=touchpoint_id,
            episode_id=episode_id,
            user_id=user_id,
            campaign_id=campaign_id,
            timestamp=datetime.now(),
            action=action,
            state=state,
            immediate_reward=immediate_reward,
            channel=kwargs.get('channel', 'unknown'),
            creative_type=kwargs.get('creative_type', 'unknown'),
            placement=kwargs.get('placement', 'unknown'),
            position_in_journey=len(self.user_journeys[user_id]),
            session_id=kwargs.get('session_id'),
            cost=kwargs.get('cost', 0.0),
            metadata=kwargs.get('metadata', {})
        )
        
        # Add to user journey
        self.user_journeys[user_id].append(touchpoint)
        self.episode_touchpoints[episode_id].append(touchpoint_id)
        
        # Create pending reward
        pending_reward = PendingReward(
            reward_id=str(uuid.uuid4()),
            touchpoints=[touchpoint],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=self.config.attribution_window_days),
            attribution_model=self.config.default_attribution_model
        )
        
        self.pending_rewards[pending_reward.reward_id] = pending_reward
        
        # Store in persistent storage
        await self._persist_touchpoint(touchpoint)
        
        # Cache in Redis
        if self.redis_client:
            await self._cache_pending_reward(pending_reward)
        
        self.logger.debug(f"Stored pending reward {touchpoint_id} for user {user_id}")
        
        return touchpoint_id
    
    async def trigger_attribution(self,
                                 user_id: str,
                                 conversion_event: ConversionEvent,
                                 conversion_value: float,
                                 **kwargs) -> Dict[str, float]:
        """
        Trigger attribution when a conversion is detected
        
        Args:
            user_id: User who converted
            conversion_event: Type of conversion
            conversion_value: Value of conversion
            **kwargs: Additional conversion metadata
            
        Returns:
            Dictionary mapping touchpoint_ids to attributed rewards
        """
        
        # Create conversion record
        conversion = ConversionRecord(
            conversion_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.now(),
            event_type=conversion_event,
            value=conversion_value,
            currency=kwargs.get('currency', 'USD'),
            product_id=kwargs.get('product_id'),
            category=kwargs.get('category'),
            metadata=kwargs.get('metadata', {})
        )
        
        # Store conversion
        await self._persist_conversion(conversion)
        
        # Calculate dynamic attribution window if enabled
        attribution_window_days = self.config.attribution_window_days
        if (self.config.dynamic_attribution_windows and 
            self.conversion_lag_model and 
            self.conversion_lag_model.is_fitted):
            attribution_window_days = await self.calculate_dynamic_attribution_window(user_id)
        
        # Get touchpoints within attribution window
        attribution_window_start = conversion.timestamp - timedelta(days=attribution_window_days)
        relevant_touchpoints = [
            tp for tp in self.user_journeys.get(user_id, [])
            if tp.timestamp >= attribution_window_start and tp.timestamp <= conversion.timestamp
        ]
        
        if not relevant_touchpoints:
            self.logger.info(f"No touchpoints found for user {user_id} within attribution window")
            return {}
        
        # Apply attribution model
        attribution_rewards = await self._apply_attribution_model(
            relevant_touchpoints, 
            conversion,
            self.config.default_attribution_model
        )
        
        # Update replay buffer with attributed rewards
        for touchpoint in relevant_touchpoints:
            attributed_reward = attribution_rewards.get(touchpoint.touchpoint_id, 0.0)
            if attributed_reward != touchpoint.immediate_reward:
                self.replay_buffer.add_experience(
                    touchpoint.episode_id,
                    touchpoint.immediate_reward,
                    attributed_reward,
                    touchpoint,
                    conversion,
                    priority=abs(attributed_reward - touchpoint.immediate_reward)
                )
        
        # Update statistics
        self.attribution_stats['total_conversions_attributed'] += 1
        self.attribution_stats['total_reward_attributed'] += sum(attribution_rewards.values())
        
        self.logger.info(f"Attributed conversion {conversion.conversion_id} to {len(attribution_rewards)} touchpoints")
        
        return attribution_rewards
    
    async def handle_partial_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        """
        Handle partial episodes where conversions happen after episode ends
        
        Args:
            episode_id: Episode that ended before conversions
            
        Returns:
            List of delayed reward updates for the episode
        """
        
        delayed_rewards = []
        
        # Get touchpoints from the episode
        touchpoint_ids = self.episode_touchpoints.get(episode_id, [])
        
        for touchpoint_id in touchpoint_ids:
            # Check if this touchpoint has been attributed
            for pending_reward in self.pending_rewards.values():
                for touchpoint in pending_reward.touchpoints:
                    if (touchpoint.touchpoint_id == touchpoint_id and 
                        touchpoint.episode_id == episode_id and
                        pending_reward.status == "attributed"):
                        
                        attributed_reward = pending_reward.attribution_rewards.get(touchpoint_id, 0.0)
                        
                        delayed_rewards.append({
                            'touchpoint_id': touchpoint_id,
                            'original_reward': touchpoint.immediate_reward,
                            'attributed_reward': attributed_reward,
                            'reward_delta': attributed_reward - touchpoint.immediate_reward,
                            'attribution_model': pending_reward.attribution_model.value,
                            'attribution_timestamp': datetime.now()
                        })
        
        return delayed_rewards
    
    async def _apply_attribution_model(self,
                                     touchpoints: List[Touchpoint],
                                     conversion: ConversionRecord,
                                     model: AttributionModel) -> Dict[str, float]:
        """Apply selected attribution model to distribute conversion value"""
        
        if not touchpoints:
            return {}
        
        total_value = conversion.value
        attribution_rewards = {}
        
        if model == AttributionModel.LAST_CLICK:
            # All credit to last touchpoint
            last_touchpoint = max(touchpoints, key=lambda t: t.timestamp)
            attribution_rewards[last_touchpoint.touchpoint_id] = total_value
            
        elif model == AttributionModel.FIRST_CLICK:
            # All credit to first touchpoint
            first_touchpoint = min(touchpoints, key=lambda t: t.timestamp)
            attribution_rewards[first_touchpoint.touchpoint_id] = total_value
            
        elif model == AttributionModel.LINEAR:
            # Equal credit to all touchpoints
            credit_per_touchpoint = total_value / len(touchpoints)
            for touchpoint in touchpoints:
                attribution_rewards[touchpoint.touchpoint_id] = credit_per_touchpoint
                
        elif model == AttributionModel.TIME_DECAY:
            # Exponential decay based on time to conversion
            weights = []
            half_life = self.config.time_decay_half_life_hours * 3600  # Convert to seconds
            
            for touchpoint in touchpoints:
                time_to_conversion = (conversion.timestamp - touchpoint.timestamp).total_seconds()
                weight = 2 ** (-time_to_conversion / half_life)
                weights.append(weight)
            
            total_weight = sum(weights)
            for touchpoint, weight in zip(touchpoints, weights):
                attribution_rewards[touchpoint.touchpoint_id] = (weight / total_weight) * total_value
                
        elif model == AttributionModel.POSITION_BASED:
            # 40% to first, 40% to last, 20% distributed among others
            if len(touchpoints) == 1:
                attribution_rewards[touchpoints[0].touchpoint_id] = total_value
            elif len(touchpoints) == 2:
                first_touchpoint = min(touchpoints, key=lambda t: t.timestamp)
                last_touchpoint = max(touchpoints, key=lambda t: t.timestamp)
                attribution_rewards[first_touchpoint.touchpoint_id] = total_value * 0.5
                attribution_rewards[last_touchpoint.touchpoint_id] = total_value * 0.5
            else:
                sorted_touchpoints = sorted(touchpoints, key=lambda t: t.timestamp)
                first_credit = total_value * self.config.position_based_first_weight
                last_credit = total_value * self.config.position_based_last_weight
                middle_credit = total_value * (1 - self.config.position_based_first_weight - self.config.position_based_last_weight)
                
                attribution_rewards[sorted_touchpoints[0].touchpoint_id] = first_credit
                attribution_rewards[sorted_touchpoints[-1].touchpoint_id] = last_credit
                
                middle_touchpoints = sorted_touchpoints[1:-1]
                if middle_touchpoints:
                    credit_per_middle = middle_credit / len(middle_touchpoints)
                    for touchpoint in middle_touchpoints:
                        attribution_rewards[touchpoint.touchpoint_id] = credit_per_middle
                        
        elif model == AttributionModel.DATA_DRIVEN:
            # Simple data-driven model based on historical conversion rates
            weights = []
            for touchpoint in touchpoints:
                # Weight based on channel, creative type, and position
                base_weight = 1.0
                
                # Channel weights (example values)
                channel_weights = {
                    'search': 1.5,
                    'display': 0.8,
                    'social': 1.2,
                    'email': 1.0,
                    'direct': 1.8
                }
                base_weight *= channel_weights.get(touchpoint.channel, 1.0)
                
                # Position in journey weight
                position_weight = 1.0 + (touchpoint.position_in_journey * 0.1)
                base_weight *= position_weight
                
                weights.append(base_weight)
            
            total_weight = sum(weights)
            for touchpoint, weight in zip(touchpoints, weights):
                attribution_rewards[touchpoint.touchpoint_id] = (weight / total_weight) * total_value
        
        return attribution_rewards
    
    async def get_replay_batch(self, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get a batch of experiences for training with delayed rewards"""
        
        if batch_size is None:
            batch_size = self.config.replay_batch_size
            
        if len(self.replay_buffer.buffer) < self.config.min_replay_samples:
            return []
        
        return self.replay_buffer.sample_batch(batch_size)
    
    async def _persist_touchpoint(self, touchpoint: Touchpoint):
        """Persist touchpoint to database"""
        
        if not self.db_session:
            return
            
        try:
            touchpoint_record = TouchpointTable(
                touchpoint_id=touchpoint.touchpoint_id,
                episode_id=touchpoint.episode_id,
                user_id=touchpoint.user_id,
                campaign_id=touchpoint.campaign_id,
                timestamp=touchpoint.timestamp,
                action_data=pickle.dumps(touchpoint.action).hex(),
                state_data=pickle.dumps(touchpoint.state).hex(),
                immediate_reward=touchpoint.immediate_reward,
                channel=touchpoint.channel,
                creative_type=touchpoint.creative_type,
                placement=touchpoint.placement,
                position_in_journey=touchpoint.position_in_journey,
                session_id=touchpoint.session_id,
                cost=touchpoint.cost,
                metadata_json=pickle.dumps(touchpoint.metadata).hex()
            )
            
            self.db_session.add(touchpoint_record)
            self.db_session.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to persist touchpoint: {e}")
            self.db_session.rollback()
    
    async def _persist_conversion(self, conversion: ConversionRecord):
        """Persist conversion to database"""
        
        if not self.db_session:
            return
            
        try:
            conversion_record = ConversionTable(
                conversion_id=conversion.conversion_id,
                user_id=conversion.user_id,
                timestamp=conversion.timestamp,
                event_type=conversion.event_type.value,
                value=conversion.value,
                currency=conversion.currency,
                product_id=conversion.product_id,
                category=conversion.category,
                metadata_json=pickle.dumps(conversion.metadata).hex()
            )
            
            self.db_session.add(conversion_record)
            self.db_session.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to persist conversion: {e}")
            self.db_session.rollback()
    
    async def _cache_pending_reward(self, pending_reward: PendingReward):
        """Cache pending reward in Redis"""
        
        if not self.redis_client:
            return
            
        try:
            key = f"pending_reward:{pending_reward.reward_id}"
            data = pickle.dumps(pending_reward)
            self.redis_client.setex(key, self.config.redis_ttl_seconds, data.hex())
            
        except Exception as e:
            self.logger.error(f"Failed to cache pending reward: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired pending rewards"""
        
        while True:
            try:
                current_time = datetime.now()
                expired_rewards = []
                
                for reward_id, pending_reward in self.pending_rewards.items():
                    if pending_reward.expires_at and pending_reward.expires_at < current_time:
                        expired_rewards.append(reward_id)
                
                # Remove expired rewards
                for reward_id in expired_rewards:
                    del self.pending_rewards[reward_id]
                    
                    # Remove from Redis cache
                    if self.redis_client:
                        self.redis_client.delete(f"pending_reward:{reward_id}")
                
                self.logger.info(f"Cleaned up {len(expired_rewards)} expired pending rewards")
                
                # Update statistics
                self.attribution_stats['pending_rewards_count'] = len(self.pending_rewards)
                
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)  # Wait an hour before retrying
    
    async def _periodic_attribution(self):
        """Periodic attribution processing for batch efficiency"""
        
        while True:
            try:
                # Process pending attributions in batches
                await self._process_attribution_batch()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Attribution task error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes before retrying
    
    async def _process_attribution_batch(self):
        """Process a batch of pending attributions"""
        
        # This would typically check for new conversions from external systems
        # and trigger attribution for relevant pending rewards
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        return {
            'attribution_stats': self.attribution_stats.copy(),
            'pending_rewards': len(self.pending_rewards),
            'user_journeys': len(self.user_journeys),
            'total_touchpoints': sum(len(journey) for journey in self.user_journeys.values()),
            'replay_buffer': self.replay_buffer.get_statistics(),
            'storage': {
                'redis_connected': self.redis_client is not None,
                'database_connected': self.db_session is not None
            }
        }
    
    def get_user_journey(self, user_id: str) -> List[Touchpoint]:
        """Get complete user journey"""
        return self.user_journeys.get(user_id, []).copy()
    
    def get_episode_touchpoints(self, episode_id: str) -> List[str]:
        """Get touchpoint IDs for an episode"""
        return self.episode_touchpoints.get(episode_id, []).copy()
    
    async def shutdown(self):
        """Graceful shutdown of the delayed reward system"""
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.attribution_task:
            self.attribution_task.cancel()
        
        # Close database connection
        if self.db_session:
            self.db_session.close()
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        self.logger.info("Delayed reward system shutdown complete")
    
    async def calculate_dynamic_attribution_window(self, user_id: str) -> int:
        """
        Calculate dynamic attribution window based on conversion lag predictions.
        
        Args:
            user_id: User identifier
            
        Returns:
            Recommended attribution window in days
        """
        if (not self.conversion_lag_model or 
            not self.conversion_lag_model.is_fitted or
            not self.config.dynamic_attribution_windows):
            return self.config.attribution_window_days
        
        # Get user journey
        touchpoints = self.user_journeys.get(user_id, [])
        if not touchpoints:
            return self.config.attribution_window_days
        
        try:
            # Create ConversionJourney for prediction
            conversion_journey = self._create_conversion_journey(user_id, touchpoints)
            
            # Get conversion time predictions
            predictions = self.conversion_lag_model.predict_conversion_time([conversion_journey])
            conversion_probs = predictions.get(user_id, np.array([]))
            
            if len(conversion_probs) == 0:
                return self.config.attribution_window_days
            
            # Find optimal window where 95% of conversion probability is reached
            max_prob = np.max(conversion_probs)
            target_prob = max_prob * 0.95
            
            optimal_window = self.config.attribution_window_days
            for day, prob in enumerate(conversion_probs, 1):
                if prob >= target_prob:
                    optimal_window = min(day + 3, 60)  # Add 3-day buffer, max 60 days
                    break
            
            self.attribution_stats['dynamic_windows_used'] += 1
            current_avg = self.attribution_stats.get('avg_dynamic_window_days', 0.0)
            total_windows = self.attribution_stats['dynamic_windows_used']
            self.attribution_stats['avg_dynamic_window_days'] = (
                (current_avg * (total_windows - 1) + optimal_window) / total_windows
            )
            
            self.logger.debug(f"Dynamic attribution window for user {user_id}: {optimal_window} days")
            return optimal_window
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic attribution window: {e}")
            return self.config.attribution_window_days
    
    def _create_conversion_journey(self, user_id: str, touchpoints: List[Touchpoint]) -> ConversionJourney:
        """
        Create ConversionJourney object from user touchpoints.
        
        Args:
            user_id: User identifier
            touchpoints: List of user touchpoints
            
        Returns:
            ConversionJourney object
        """
        if not touchpoints:
            raise ValueError("Cannot create conversion journey without touchpoints")
        
        # Convert touchpoints to the format expected by ConversionJourney
        touchpoints_data = []
        for tp in touchpoints:
            touchpoints_data.append({
                'timestamp': tp.timestamp,
                'channel': tp.channel,
                'action': tp.action,
                'metadata': tp.metadata
            })
        
        # Extract features
        features = {
            'touchpoint_count': len(touchpoints),
            'total_cost': sum(tp.cost for tp in touchpoints),
            'channel_diversity': len(set(tp.channel for tp in touchpoints)),
            'avg_immediate_reward': np.mean([tp.immediate_reward for tp in touchpoints])
        }
        
        return ConversionJourney(
            user_id=user_id,
            start_time=touchpoints[0].timestamp,
            touchpoints=touchpoints_data,
            features=features
        )
    
    async def predict_conversion_timing(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Predict conversion timing for a user based on their journey.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with conversion timing predictions or None if not available
        """
        if not self.conversion_lag_model or not self.conversion_lag_model.is_fitted:
            return None
        
        touchpoints = self.user_journeys.get(user_id, [])
        if not touchpoints:
            return None
        
        try:
            # Create conversion journey
            conversion_journey = self._create_conversion_journey(user_id, touchpoints)
            
            # Get predictions
            predictions = self.conversion_lag_model.predict_conversion_time([conversion_journey])
            hazard_rates = self.conversion_lag_model.calculate_hazard_rate([conversion_journey])
            
            conversion_probs = predictions.get(user_id, np.array([]))
            hazard_data = hazard_rates.get(user_id, np.array([]))
            
            if len(conversion_probs) == 0:
                return None
            
            # Calculate key metrics
            peak_conversion_day = np.argmax(hazard_data) + 1 if len(hazard_data) > 0 else 1
            median_conversion_day = None
            
            # Find median (50% probability)
            for day, prob in enumerate(conversion_probs, 1):
                if prob >= 0.5:
                    median_conversion_day = day
                    break
            
            return {
                'user_id': user_id,
                'peak_conversion_day': int(peak_conversion_day),
                'median_conversion_day': int(median_conversion_day) if median_conversion_day else None,
                'max_conversion_probability': float(np.max(conversion_probs)),
                'conversion_probabilities_7_days': conversion_probs[:7].tolist() if len(conversion_probs) >= 7 else conversion_probs.tolist(),
                'conversion_probabilities_30_days': conversion_probs[:30].tolist() if len(conversion_probs) >= 30 else conversion_probs.tolist(),
                'recommended_attribution_window': await self.calculate_dynamic_attribution_window(user_id),
                'prediction_timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting conversion timing for user {user_id}: {e}")
            return None
    
    async def train_conversion_lag_model(self, 
                                       lookback_days: int = 90,
                                       min_journeys: int = 100) -> bool:
        """
        Train the conversion lag model with historical journey data.
        
        Args:
            lookback_days: Days to look back for training data
            min_journeys: Minimum number of journeys required for training
            
        Returns:
            True if training was successful, False otherwise
        """
        if not self.conversion_lag_model:
            self.logger.error("Conversion lag model not initialized")
            return False
        
        try:
            # Collect training data from user journeys and conversion records
            training_journeys = []
            
            # Convert current user journeys to ConversionJourney objects
            for user_id, touchpoints in self.user_journeys.items():
                if not touchpoints:
                    continue
                
                try:
                    journey = self._create_conversion_journey(user_id, touchpoints)
                    
                    # Check if we have conversion data for this user
                    # This would typically come from external conversion tracking
                    # For now, we'll mark as unconverted unless we have specific data
                    journey.converted = False  # Default assumption
                    
                    training_journeys.append(journey)
                    
                except Exception as e:
                    self.logger.warning(f"Error creating training journey for user {user_id}: {e}")
                    continue
            
            if len(training_journeys) < min_journeys:
                self.logger.warning(f"Insufficient training data: {len(training_journeys)} < {min_journeys}")
                return False
            
            # Train the model
            self.logger.info(f"Training conversion lag model with {len(training_journeys)} journeys")
            self.conversion_lag_model.fit(training_journeys)
            
            # Generate and log insights
            insights = self.conversion_lag_model.get_conversion_insights(training_journeys)
            self.logger.info(f"Conversion lag model training completed. Insights: {insights}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training conversion lag model: {e}")
            return False
    
    async def handle_censored_data_update(self) -> Dict[str, Any]:
        """
        Handle right-censored data by updating ongoing journeys.
        
        Returns:
            Statistics about censored data handling
        """
        if not self.conversion_lag_model:
            return {'error': 'Conversion lag model not initialized'}
        
        try:
            # Get all cached conversion journeys
            cached_journeys = []
            for user_id, touchpoints in self.user_journeys.items():
                if touchpoints:
                    journey = self._create_conversion_journey(user_id, touchpoints)
                    cached_journeys.append(journey)
            
            if not cached_journeys:
                return {'message': 'No journeys to process'}
            
            # Process censored data
            processed_journeys = self.conversion_lag_model.handle_censored_data(cached_journeys)
            
            # Update cache
            for journey in processed_journeys:
                self.conversion_journey_cache[journey.user_id] = journey
            
            stats = {
                'total_journeys_processed': len(processed_journeys),
                'censored_journeys': len([j for j in processed_journeys if j.is_censored]),
                'timeout_journeys': len([j for j in processed_journeys if j.timeout_reason == 'abandoned']),
                'processing_timestamp': datetime.now()
            }
            
            self.logger.info(f"Processed censored journey data: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error handling censored data: {e}")
            return {'error': str(e)}


# Integration helper functions for training orchestrator

async def integrate_with_episode_manager(episode_manager, delayed_reward_system: DelayedRewardSystem):
    """
    Integrate delayed reward system with episode manager
    
    This function modifies the episode manager to store pending rewards
    and handle delayed attribution during training.
    """
    
    original_execute_episode = episode_manager._execute_episode
    
    async def enhanced_execute_episode(self, agent, environment, config, result):
        """Enhanced episode execution with delayed reward tracking"""
        
        # Execute original episode logic
        await original_execute_episode(agent, environment, config, result)
        
        # Check for delayed reward updates for this episode
        delayed_updates = await delayed_reward_system.handle_partial_episode(result.episode_id)
        
        if delayed_updates:
            # Update episode result with delayed rewards
            total_delayed_reward = sum(update['attributed_reward'] for update in delayed_updates)
            result.metrics.total_reward += total_delayed_reward - sum(update['original_reward'] for update in delayed_updates)
            
            # Store delayed reward information
            result.info['delayed_rewards'] = delayed_updates
            result.info['total_delayed_reward_adjustment'] = total_delayed_reward
    
    # Replace the method
    episode_manager._execute_episode = enhanced_execute_episode.__get__(episode_manager, type(episode_manager))
    
    return episode_manager


async def create_delayed_reward_training_loop(agent, environment, delayed_reward_system: DelayedRewardSystem):
    """
    Create a training loop that incorporates delayed reward learning
    """
    
    episode_count = 0
    
    while True:
        # Regular episode
        state = await environment.reset()
        done = False
        episode_rewards = []
        episode_touchpoints = []
        
        while not done:
            action = await agent.select_action(state)
            next_state, reward, done, info = await environment.step(action)
            
            # Store pending reward
            touchpoint_id = await delayed_reward_system.store_pending_reward(
                episode_id=f"episode_{episode_count}",
                user_id=info.get('user_id', 'unknown'),
                campaign_id=info.get('campaign_id', 'unknown'),
                action=action,
                state=state,
                immediate_reward=reward,
                **info
            )
            
            episode_rewards.append(reward)
            episode_touchpoints.append(touchpoint_id)
            
            # Train agent with immediate reward
            await agent.update(state, action, reward, next_state, done)
            
            state = next_state
        
        episode_count += 1
        
        # Periodically train with delayed rewards
        if episode_count % delayed_reward_system.config.replay_frequency == 0:
            replay_batch = await delayed_reward_system.get_replay_batch()
            
            if replay_batch:
                # Train agent with corrected rewards from delayed attribution
                for experience in replay_batch:
                    await agent.update_with_corrected_reward(
                        state=experience['state'],
                        action=experience['action'],
                        corrected_reward=experience['attributed_reward'],
                        original_reward=experience['original_reward']
                    )