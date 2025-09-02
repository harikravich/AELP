"""
User Journey Tracker for Delayed Reward Attribution

Implements proper delayed reward attribution with configurable attribution windows:
- First touch: 14 days
- Last touch: 3 days  
- Multi-touch: weighted by time decay
- Handles sparse rewards (conversions days after clicks)
- Tracks user journeys across multiple touchpoints
- NO immediate-only rewards - everything uses delayed attribution

CRITICAL: This module handles delayed rewards ONLY. No fallbacks to immediate rewards.
"""

import asyncio
import logging
import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque
import uuid
import numpy as np
import threading
import pickle
from contextlib import contextmanager

# Import attribution models
from attribution_models import (
    AttributionEngine, 
    Journey, 
    Touchpoint as AttributionTouchpoint,
    TimeDecayAttribution,
    PositionBasedAttribution,
    LinearAttribution
)

logger = logging.getLogger(__name__)


class AttributionWindow(Enum):
    """Attribution window types with specific day ranges"""
    FIRST_TOUCH = 14    # 14 days for first-touch attribution  
    LAST_TOUCH = 3      # 3 days for last-touch attribution
    MULTI_TOUCH = 7     # 7 days default for multi-touch attribution
    VIEW_THROUGH = 1    # 1 day for view-through conversions
    EXTENDED = 30       # 30 days for high-value conversions


class TouchpointType(Enum):
    """Types of marketing touchpoints"""
    IMPRESSION = "impression"
    CLICK = "click"
    VIEW = "view"
    ENGAGEMENT = "engagement"
    VISIT = "visit"
    INTERACTION = "interaction"


class ConversionType(Enum):
    """Types of conversion events"""
    PURCHASE = "purchase"
    SIGNUP = "signup"
    TRIAL = "trial"
    DOWNLOAD = "download"
    LEAD = "lead"
    VIEW_CONTENT = "view_content"
    ADD_TO_CART = "add_to_cart"
    INITIATE_CHECKOUT = "initiate_checkout"


@dataclass
class JourneyTouchpoint:
    """Single touchpoint in a user journey"""
    touchpoint_id: str
    user_id: str
    timestamp: datetime
    channel: str
    touchpoint_type: TouchpointType
    campaign_id: str
    creative_id: str
    placement_id: str
    bid_amount: float
    cost: float
    immediate_reward: float  # Always 0.0 - no immediate rewards
    state_data: Dict[str, Any]
    action_data: Dict[str, Any]
    session_id: Optional[str] = None
    device_type: Optional[str] = None
    geo_location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # CRITICAL: No immediate rewards allowed
        if self.immediate_reward != 0.0:
            raise ValueError("NO IMMEDIATE REWARDS ALLOWED - Use delayed attribution only")
    
    def to_attribution_touchpoint(self) -> AttributionTouchpoint:
        """Convert to attribution model touchpoint"""
        return AttributionTouchpoint(
            id=self.touchpoint_id,
            timestamp=self.timestamp,
            channel=self.channel,
            action=self.touchpoint_type.value,
            value=self.cost,
            metadata={
                'campaign_id': self.campaign_id,
                'creative_id': self.creative_id,
                'placement_id': self.placement_id,
                'bid_amount': self.bid_amount,
                'session_id': self.session_id,
                'device_type': self.device_type,
                'geo_location': self.geo_location,
                **self.metadata
            }
        )


@dataclass
class ConversionEvent:
    """Conversion event with attribution details"""
    conversion_id: str
    user_id: str
    timestamp: datetime
    conversion_type: ConversionType
    value: float
    currency: str = "USD"
    product_id: Optional[str] = None
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DelayedReward:
    """Delayed reward attribution result"""
    reward_id: str
    user_id: str
    conversion_event: ConversionEvent
    attributed_touchpoints: List[Tuple[JourneyTouchpoint, float]]
    attribution_window_days: int
    attribution_model: str
    total_attributed_value: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserJourneyState:
    """Current state of a user's journey"""
    user_id: str
    touchpoints: List[JourneyTouchpoint] = field(default_factory=list)
    last_activity: Optional[datetime] = None
    total_touchpoints: int = 0
    total_cost: float = 0.0
    conversions: List[ConversionEvent] = field(default_factory=list)
    pending_attributions: List[str] = field(default_factory=list)  # reward_ids
    
    @property
    def journey_length_days(self) -> float:
        """Get journey length in days"""
        if not self.touchpoints:
            return 0.0
        first_touch = min(tp.timestamp for tp in self.touchpoints)
        last_touch = max(tp.timestamp for tp in self.touchpoints)
        return (last_touch - first_touch).days
    
    @property
    def unique_channels(self) -> Set[str]:
        """Get unique channels in journey"""
        return set(tp.channel for tp in self.touchpoints)
    
    @property
    def unique_campaigns(self) -> Set[str]:
        """Get unique campaigns in journey"""
        return set(tp.campaign_id for tp in self.touchpoints)


class UserJourneyTracker:
    """
    Tracks user journeys and handles delayed reward attribution
    
    CRITICAL FEATURES:
    - NO immediate rewards - everything is delayed
    - Multi-day attribution windows (3-14 days)
    - Multi-touch attribution with time decay
    - Proper handling of sparse rewards
    - Persistent storage of journey data
    """
    
    def __init__(self, db_path: str = "user_journeys.db"):
        self.db_path = db_path
        self.user_journeys: Dict[str, UserJourneyState] = {}
        self.pending_rewards: Dict[str, DelayedReward] = {}
        self.attribution_engine = AttributionEngine()
        self.lock = threading.RLock()
        
        # Attribution window configurations
        self.attribution_windows = {
            'first_touch': AttributionWindow.FIRST_TOUCH.value,
            'last_touch': AttributionWindow.LAST_TOUCH.value,
            'multi_touch': AttributionWindow.MULTI_TOUCH.value,
            'view_through': AttributionWindow.VIEW_THROUGH.value,
            'extended': AttributionWindow.EXTENDED.value
        }
        
        # Time decay configuration for multi-touch attribution
        self.time_decay_half_life_days = 7.0
        
        # Initialize database
        self._init_database()
        self._load_existing_journeys()
        
        logger.info("UserJourneyTracker initialized with delayed reward attribution")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS touchpoints (
                    touchpoint_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    touchpoint_type TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    creative_id TEXT NOT NULL,
                    placement_id TEXT NOT NULL,
                    bid_amount REAL NOT NULL,
                    cost REAL NOT NULL,
                    immediate_reward REAL NOT NULL CHECK(immediate_reward = 0.0),
                    state_data TEXT NOT NULL,
                    action_data TEXT NOT NULL,
                    session_id TEXT,
                    device_type TEXT,
                    geo_location TEXT,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversions (
                    conversion_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    conversion_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    currency TEXT NOT NULL,
                    product_id TEXT,
                    category TEXT,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS delayed_rewards (
                    reward_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    conversion_id TEXT NOT NULL,
                    attribution_window_days INTEGER NOT NULL,
                    attribution_model TEXT NOT NULL,
                    total_attributed_value REAL NOT NULL,
                    touchpoint_attributions TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_touchpoints_user_time ON touchpoints(user_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_conversions_user_time ON conversions(user_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_rewards_user ON delayed_rewards(user_id)')
            
            conn.commit()
    
    def _load_existing_journeys(self):
        """Load existing user journeys from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Load touchpoints
                touchpoints_cursor = conn.execute('''
                    SELECT * FROM touchpoints ORDER BY user_id, timestamp
                ''')
                
                for row in touchpoints_cursor:
                    touchpoint = JourneyTouchpoint(
                        touchpoint_id=row['touchpoint_id'],
                        user_id=row['user_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        channel=row['channel'],
                        touchpoint_type=TouchpointType(row['touchpoint_type']),
                        campaign_id=row['campaign_id'],
                        creative_id=row['creative_id'],
                        placement_id=row['placement_id'],
                        bid_amount=row['bid_amount'],
                        cost=row['cost'],
                        immediate_reward=0.0,  # Always 0 - enforced by DB constraint
                        state_data=json.loads(row['state_data']),
                        action_data=json.loads(row['action_data']),
                        session_id=row['session_id'],
                        device_type=row['device_type'],
                        geo_location=row['geo_location'],
                        metadata=json.loads(row['metadata'])
                    )
                    
                    if touchpoint.user_id not in self.user_journeys:
                        self.user_journeys[touchpoint.user_id] = UserJourneyState(
                            user_id=touchpoint.user_id
                        )
                    
                    self.user_journeys[touchpoint.user_id].touchpoints.append(touchpoint)
                    self.user_journeys[touchpoint.user_id].last_activity = max(
                        self.user_journeys[touchpoint.user_id].last_activity or touchpoint.timestamp,
                        touchpoint.timestamp
                    )
                    self.user_journeys[touchpoint.user_id].total_touchpoints += 1
                    self.user_journeys[touchpoint.user_id].total_cost += touchpoint.cost
                
                # Load conversions
                conversions_cursor = conn.execute('''
                    SELECT * FROM conversions ORDER BY user_id, timestamp
                ''')
                
                for row in conversions_cursor:
                    conversion = ConversionEvent(
                        conversion_id=row['conversion_id'],
                        user_id=row['user_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        conversion_type=ConversionType(row['conversion_type']),
                        value=row['value'],
                        currency=row['currency'],
                        product_id=row['product_id'],
                        category=row['category'],
                        metadata=json.loads(row['metadata'])
                    )
                    
                    if conversion.user_id in self.user_journeys:
                        self.user_journeys[conversion.user_id].conversions.append(conversion)
                
                # Load delayed rewards
                rewards_cursor = conn.execute('''
                    SELECT * FROM delayed_rewards ORDER BY user_id, created_at
                ''')
                
                for row in rewards_cursor:
                    # Find the conversion event
                    conversion_event = None
                    for user_journey in self.user_journeys.values():
                        for conversion in user_journey.conversions:
                            if conversion.conversion_id == row['conversion_id']:
                                conversion_event = conversion
                                break
                        if conversion_event:
                            break
                    
                    if not conversion_event:
                        continue  # Skip if we can't find the conversion
                    
                    # Parse touchpoint attributions
                    touchpoint_attributions_data = json.loads(row['touchpoint_attributions'])
                    attributed_touchpoints = []
                    
                    for tp_attr in touchpoint_attributions_data:
                        # Find the touchpoint
                        touchpoint = None
                        if row['user_id'] in self.user_journeys:
                            for tp in self.user_journeys[row['user_id']].touchpoints:
                                if tp.touchpoint_id == tp_attr['touchpoint_id']:
                                    touchpoint = tp
                                    break
                        
                        if touchpoint:
                            attributed_touchpoints.append((touchpoint, tp_attr['attributed_credit']))
                    
                    if attributed_touchpoints:
                        # Recreate delayed reward
                        delayed_reward = DelayedReward(
                            reward_id=row['reward_id'],
                            user_id=row['user_id'],
                            conversion_event=conversion_event,
                            attributed_touchpoints=attributed_touchpoints,
                            attribution_window_days=row['attribution_window_days'],
                            attribution_model=row['attribution_model'],
                            total_attributed_value=row['total_attributed_value'],
                            created_at=datetime.fromisoformat(row['created_at'])
                        )
                        
                        self.pending_rewards[delayed_reward.reward_id] = delayed_reward
                        
                        # Add to user journey pending attributions
                        if row['user_id'] in self.user_journeys:
                            self.user_journeys[row['user_id']].pending_attributions.append(delayed_reward.reward_id)
            
            logger.info(f"Loaded {len(self.user_journeys)} user journeys and {len(self.pending_rewards)} delayed rewards from database")
            
        except sqlite3.OperationalError as e:
            # Tables don't exist yet - this is fine for new databases
            logger.info(f"No existing journey data found (new database): {e}")
        except Exception as e:
            logger.warning(f"Error loading existing journeys: {e}")
    
    def add_touchpoint(self,
                      user_id: str,
                      channel: str,
                      touchpoint_type: TouchpointType,
                      campaign_id: str,
                      creative_id: str,
                      placement_id: str,
                      bid_amount: float,
                      cost: float,
                      state_data: Dict[str, Any],
                      action_data: Dict[str, Any],
                      session_id: Optional[str] = None,
                      device_type: Optional[str] = None,
                      geo_location: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new touchpoint to user journey
        
        CRITICAL: immediate_reward is ALWAYS 0.0 - no immediate rewards allowed
        
        Returns:
            touchpoint_id of the created touchpoint
        """
        with self.lock:
            touchpoint_id = str(uuid.uuid4())
            
            touchpoint = JourneyTouchpoint(
                touchpoint_id=touchpoint_id,
                user_id=user_id,
                timestamp=datetime.now(),
                channel=channel,
                touchpoint_type=touchpoint_type,
                campaign_id=campaign_id,
                creative_id=creative_id,
                placement_id=placement_id,
                bid_amount=bid_amount,
                cost=cost,
                immediate_reward=0.0,  # ALWAYS 0.0 - no immediate rewards
                state_data=state_data,
                action_data=action_data,
                session_id=session_id,
                device_type=device_type,
                geo_location=geo_location,
                metadata=metadata or {}
            )
            
            # Initialize user journey if doesn't exist
            if user_id not in self.user_journeys:
                self.user_journeys[user_id] = UserJourneyState(user_id=user_id)
            
            # Add touchpoint to journey
            self.user_journeys[user_id].touchpoints.append(touchpoint)
            self.user_journeys[user_id].last_activity = touchpoint.timestamp
            self.user_journeys[user_id].total_touchpoints += 1
            self.user_journeys[user_id].total_cost += cost
            
            # Persist to database
            self._persist_touchpoint(touchpoint)
            
            logger.debug(f"Added touchpoint {touchpoint_id} for user {user_id} - NO immediate reward")
            return touchpoint_id
    
    def record_conversion(self,
                         user_id: str,
                         conversion_type: ConversionType,
                         value: float,
                         currency: str = "USD",
                         product_id: Optional[str] = None,
                         category: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> List[DelayedReward]:
        """
        Record a conversion and trigger delayed reward attribution
        
        Returns:
            List of DelayedReward objects with attribution results
        """
        with self.lock:
            conversion_id = str(uuid.uuid4())
            
            conversion = ConversionEvent(
                conversion_id=conversion_id,
                user_id=user_id,
                timestamp=datetime.now(),
                conversion_type=conversion_type,
                value=value,
                currency=currency,
                product_id=product_id,
                category=category,
                metadata=metadata or {}
            )
            
            # Add to user journey if exists
            if user_id in self.user_journeys:
                self.user_journeys[user_id].conversions.append(conversion)
            
            # Persist conversion
            self._persist_conversion(conversion)
            
            # Trigger attribution for different window types
            delayed_rewards = self._attribute_conversion(conversion)
            
            logger.info(f"Recorded conversion {conversion_id} for user {user_id}, "
                       f"attributed to {sum(len(reward.attributed_touchpoints) for reward in delayed_rewards)} touchpoints")
            
            return delayed_rewards
    
    def _attribute_conversion(self, conversion: ConversionEvent) -> List[DelayedReward]:
        """
        Attribute conversion to touchpoints using multiple attribution models and windows
        """
        if conversion.user_id not in self.user_journeys:
            logger.warning(f"No journey found for user {conversion.user_id}")
            return []
        
        user_journey = self.user_journeys[conversion.user_id]
        if not user_journey.touchpoints:
            logger.warning(f"No touchpoints found for user {conversion.user_id}")
            return []
        
        delayed_rewards = []
        
        # Apply different attribution models with their respective windows
        attribution_configs = [
            {
                'model_name': 'first_touch',
                'window_days': self.attribution_windows['first_touch'],
                'attribution_model': PositionBasedAttribution(first_weight=1.0, last_weight=0.0, middle_weight=0.0)
            },
            {
                'model_name': 'last_touch', 
                'window_days': self.attribution_windows['last_touch'],
                'attribution_model': PositionBasedAttribution(first_weight=0.0, last_weight=1.0, middle_weight=0.0)
            },
            {
                'model_name': 'multi_touch_linear',
                'window_days': self.attribution_windows['multi_touch'],
                'attribution_model': LinearAttribution()
            },
            {
                'model_name': 'multi_touch_time_decay',
                'window_days': self.attribution_windows['multi_touch'],
                'attribution_model': TimeDecayAttribution(half_life_days=int(self.time_decay_half_life_days))
            }
        ]
        
        for config in attribution_configs:
            delayed_reward = self._apply_attribution_model(conversion, config)
            if delayed_reward and delayed_reward.attributed_touchpoints:
                delayed_rewards.append(delayed_reward)
                # Persist the delayed reward
                self._persist_delayed_reward(delayed_reward)
        
        return delayed_rewards
    
    def _apply_attribution_model(self, 
                                conversion: ConversionEvent, 
                                config: Dict[str, Any]) -> Optional[DelayedReward]:
        """Apply specific attribution model with given window"""
        user_journey = self.user_journeys[conversion.user_id]
        
        # Filter touchpoints within attribution window
        window_start = conversion.timestamp - timedelta(days=config['window_days'])
        relevant_touchpoints = [
            tp for tp in user_journey.touchpoints
            if window_start <= tp.timestamp <= conversion.timestamp
        ]
        
        if not relevant_touchpoints:
            return None
        
        # Convert to attribution model format
        attribution_touchpoints = [tp.to_attribution_touchpoint() for tp in relevant_touchpoints]
        
        # Create journey for attribution
        journey = Journey(
            id=f"{conversion.user_id}_{conversion.conversion_id}",
            touchpoints=attribution_touchpoints,
            conversion_value=conversion.value,
            conversion_timestamp=conversion.timestamp,
            converted=True
        )
        
        # Calculate attribution
        attribution_model = config['attribution_model']
        attributed_credits = attribution_model.distribute_credit(journey)
        
        # Create attributed touchpoint list
        attributed_touchpoints = []
        for i, (attr_touchpoint, credit) in enumerate(attributed_credits):
            original_touchpoint = relevant_touchpoints[i]
            attributed_touchpoints.append((original_touchpoint, credit))
        
        # Create delayed reward
        reward_id = str(uuid.uuid4())
        delayed_reward = DelayedReward(
            reward_id=reward_id,
            user_id=conversion.user_id,
            conversion_event=conversion,
            attributed_touchpoints=attributed_touchpoints,
            attribution_window_days=config['window_days'],
            attribution_model=config['model_name'],
            total_attributed_value=sum(credit for _, credit in attributed_touchpoints)
        )
        
        # Add to pending rewards
        self.pending_rewards[reward_id] = delayed_reward
        
        # Add to user journey pending attributions
        user_journey.pending_attributions.append(reward_id)
        
        return delayed_reward
    
    def get_attributed_rewards_for_touchpoint(self, 
                                            touchpoint_id: str, 
                                            max_days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Get all attributed rewards for a specific touchpoint
        
        This is used to get the delayed reward signal for RL training
        """
        attributed_rewards = []
        
        cutoff_date = datetime.now() - timedelta(days=max_days_back)
        
        for reward in self.pending_rewards.values():
            if reward.created_at < cutoff_date:
                continue
                
            for touchpoint, credit in reward.attributed_touchpoints:
                if touchpoint.touchpoint_id == touchpoint_id:
                    attributed_rewards.append({
                        'reward_id': reward.reward_id,
                        'attributed_value': credit,
                        'conversion_value': reward.conversion_event.value,
                        'conversion_type': reward.conversion_event.conversion_type.value,
                        'attribution_model': reward.attribution_model,
                        'attribution_window_days': reward.attribution_window_days,
                        'time_to_conversion_hours': (
                            reward.conversion_event.timestamp - touchpoint.timestamp
                        ).total_seconds() / 3600,
                        'attribution_timestamp': reward.created_at,
                        'original_immediate_reward': 0.0,  # Always 0 - no immediate rewards
                        'reward_delta': credit  # Full credit since immediate was 0
                    })
        
        return attributed_rewards
    
    def get_user_journey(self, user_id: str) -> Optional[UserJourneyState]:
        """Get complete user journey state"""
        return self.user_journeys.get(user_id)
    
    def get_journey_statistics(self) -> Dict[str, Any]:
        """Get comprehensive journey and attribution statistics"""
        total_touchpoints = sum(len(journey.touchpoints) for journey in self.user_journeys.values())
        total_conversions = sum(len(journey.conversions) for journey in self.user_journeys.values())
        total_attributed_value = sum(reward.total_attributed_value for reward in self.pending_rewards.values())
        
        # Calculate attribution model distribution
        attribution_model_counts = defaultdict(int)
        for reward in self.pending_rewards.values():
            attribution_model_counts[reward.attribution_model] += 1
        
        # Calculate average time to conversion
        time_to_conversions = []
        for reward in self.pending_rewards.values():
            for touchpoint, _ in reward.attributed_touchpoints:
                time_to_conversion = (
                    reward.conversion_event.timestamp - touchpoint.timestamp
                ).total_seconds() / 3600
                time_to_conversions.append(time_to_conversion)
        
        return {
            'total_users': len(self.user_journeys),
            'total_touchpoints': total_touchpoints,
            'total_conversions': total_conversions,
            'total_delayed_rewards': len(self.pending_rewards),
            'total_attributed_value': total_attributed_value,
            'avg_touchpoints_per_user': total_touchpoints / len(self.user_journeys) if self.user_journeys else 0,
            'avg_time_to_conversion_hours': np.mean(time_to_conversions) if time_to_conversions else 0,
            'attribution_model_distribution': dict(attribution_model_counts),
            'attribution_windows_used': {
                name: days for name, days in self.attribution_windows.items()
            },
            'users_with_conversions': len([j for j in self.user_journeys.values() if j.conversions]),
            'conversion_rate': (
                len([j for j in self.user_journeys.values() if j.conversions]) / 
                len(self.user_journeys) if self.user_journeys else 0
            )
        }
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old journey data beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.lock:
            # Clean up old delayed rewards
            old_rewards = [
                reward_id for reward_id, reward in self.pending_rewards.items()
                if reward.created_at < cutoff_date
            ]
            
            for reward_id in old_rewards:
                del self.pending_rewards[reward_id]
            
            # Clean up old touchpoints from user journeys
            for user_journey in self.user_journeys.values():
                user_journey.touchpoints = [
                    tp for tp in user_journey.touchpoints
                    if tp.timestamp >= cutoff_date
                ]
                user_journey.conversions = [
                    conv for conv in user_journey.conversions
                    if conv.timestamp >= cutoff_date
                ]
                user_journey.pending_attributions = [
                    reward_id for reward_id in user_journey.pending_attributions
                    if reward_id not in old_rewards
                ]
            
            # Clean up database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'DELETE FROM touchpoints WHERE timestamp < ?', 
                    (cutoff_date.isoformat(),)
                )
                conn.execute(
                    'DELETE FROM conversions WHERE timestamp < ?',
                    (cutoff_date.isoformat(),)
                )
                conn.execute(
                    'DELETE FROM delayed_rewards WHERE created_at < ?',
                    (cutoff_date.isoformat(),)
                )
                conn.commit()
        
        logger.info(f"Cleaned up {len(old_rewards)} old delayed rewards and associated data")
    
    def _persist_touchpoint(self, touchpoint: JourneyTouchpoint):
        """Persist touchpoint to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO touchpoints 
                (touchpoint_id, user_id, timestamp, channel, touchpoint_type, campaign_id, 
                 creative_id, placement_id, bid_amount, cost, immediate_reward, state_data, 
                 action_data, session_id, device_type, geo_location, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                touchpoint.touchpoint_id,
                touchpoint.user_id,
                touchpoint.timestamp.isoformat(),
                touchpoint.channel,
                touchpoint.touchpoint_type.value,
                touchpoint.campaign_id,
                touchpoint.creative_id,
                touchpoint.placement_id,
                touchpoint.bid_amount,
                touchpoint.cost,
                touchpoint.immediate_reward,  # Always 0.0
                json.dumps(touchpoint.state_data),
                json.dumps(touchpoint.action_data),
                touchpoint.session_id,
                touchpoint.device_type,
                touchpoint.geo_location,
                json.dumps(touchpoint.metadata),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def _persist_conversion(self, conversion: ConversionEvent):
        """Persist conversion to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO conversions 
                (conversion_id, user_id, timestamp, conversion_type, value, currency, 
                 product_id, category, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                conversion.conversion_id,
                conversion.user_id,
                conversion.timestamp.isoformat(),
                conversion.conversion_type.value,
                conversion.value,
                conversion.currency,
                conversion.product_id,
                conversion.category,
                json.dumps(conversion.metadata),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def _persist_delayed_reward(self, delayed_reward: DelayedReward):
        """Persist delayed reward to database"""
        # Serialize touchpoint attributions
        touchpoint_attributions = []
        for touchpoint, credit in delayed_reward.attributed_touchpoints:
            touchpoint_attributions.append({
                'touchpoint_id': touchpoint.touchpoint_id,
                'attributed_credit': credit,
                'touchpoint_timestamp': touchpoint.timestamp.isoformat()
            })
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO delayed_rewards
                (reward_id, user_id, conversion_id, attribution_window_days, attribution_model,
                 total_attributed_value, touchpoint_attributions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                delayed_reward.reward_id,
                delayed_reward.user_id,
                delayed_reward.conversion_event.conversion_id,
                delayed_reward.attribution_window_days,
                delayed_reward.attribution_model,
                delayed_reward.total_attributed_value,
                json.dumps(touchpoint_attributions),
                delayed_reward.created_at.isoformat()
            ))
            conn.commit()
    
    def export_journey_data_for_training(self, 
                                       user_ids: Optional[List[str]] = None,
                                       days_back: int = 30) -> Dict[str, Any]:
        """
        Export journey data formatted for RL training with delayed rewards
        
        Args:
            user_ids: Specific users to export (None for all)
            days_back: Days of history to include
            
        Returns:
            Dictionary with training data including delayed reward signals
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        training_data = {
            'touchpoints': [],
            'delayed_rewards': [],
            'user_journeys': {},
            'export_metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'days_back': days_back,
                'total_users': 0,
                'total_touchpoints': 0,
                'total_delayed_rewards': 0
            }
        }
        
        users_to_export = user_ids if user_ids else list(self.user_journeys.keys())
        
        for user_id in users_to_export:
            if user_id not in self.user_journeys:
                continue
            
            user_journey = self.user_journeys[user_id]
            
            # Filter touchpoints by date
            relevant_touchpoints = [
                tp for tp in user_journey.touchpoints
                if tp.timestamp >= cutoff_date
            ]
            
            if not relevant_touchpoints:
                continue
            
            # Export touchpoints with state/action data for RL
            for touchpoint in relevant_touchpoints:
                training_data['touchpoints'].append({
                    'touchpoint_id': touchpoint.touchpoint_id,
                    'user_id': touchpoint.user_id,
                    'timestamp': touchpoint.timestamp.isoformat(),
                    'state': touchpoint.state_data,
                    'action': touchpoint.action_data,
                    'immediate_reward': 0.0,  # Always 0 - no immediate rewards
                    'cost': touchpoint.cost,
                    'channel': touchpoint.channel,
                    'campaign_id': touchpoint.campaign_id,
                    'metadata': touchpoint.metadata
                })
            
            # Export user journey summary
            training_data['user_journeys'][user_id] = {
                'total_touchpoints': len(relevant_touchpoints),
                'total_conversions': len(user_journey.conversions),
                'journey_length_days': user_journey.journey_length_days,
                'unique_channels': list(user_journey.unique_channels),
                'unique_campaigns': list(user_journey.unique_campaigns),
                'total_cost': user_journey.total_cost,
                'last_activity': user_journey.last_activity.isoformat() if user_journey.last_activity else None
            }
        
        # Export delayed rewards for the specified users
        for reward in self.pending_rewards.values():
            if reward.user_id in users_to_export and reward.created_at >= cutoff_date:
                reward_data = {
                    'reward_id': reward.reward_id,
                    'user_id': reward.user_id,
                    'conversion_event': {
                        'conversion_id': reward.conversion_event.conversion_id,
                        'timestamp': reward.conversion_event.timestamp.isoformat(),
                        'type': reward.conversion_event.conversion_type.value,
                        'value': reward.conversion_event.value
                    },
                    'attribution_model': reward.attribution_model,
                    'attribution_window_days': reward.attribution_window_days,
                    'total_attributed_value': reward.total_attributed_value,
                    'touchpoint_rewards': [
                        {
                            'touchpoint_id': tp.touchpoint_id,
                            'attributed_reward': credit,
                            'original_immediate_reward': 0.0,  # Always 0
                            'reward_delta': credit  # Full credit since immediate was 0
                        }
                        for tp, credit in reward.attributed_touchpoints
                    ]
                }
                training_data['delayed_rewards'].append(reward_data)
        
        # Update metadata
        training_data['export_metadata']['total_users'] = len(training_data['user_journeys'])
        training_data['export_metadata']['total_touchpoints'] = len(training_data['touchpoints'])
        training_data['export_metadata']['total_delayed_rewards'] = len(training_data['delayed_rewards'])
        
        return training_data


def verify_no_immediate_rewards():
    """
    Verification function to ensure no immediate rewards are being used
    This function should be called during testing to verify compliance
    """
    # Test that adding touchpoint with non-zero immediate reward fails
    try:
        touchpoint = JourneyTouchpoint(
            touchpoint_id="test",
            user_id="test_user",
            timestamp=datetime.now(),
            channel="search",
            touchpoint_type=TouchpointType.CLICK,
            campaign_id="test_campaign",
            creative_id="test_creative",
            placement_id="test_placement",
            bid_amount=1.0,
            cost=0.5,
            immediate_reward=1.0,  # This should fail
            state_data={},
            action_data={}
        )
        raise AssertionError("CRITICAL: Immediate reward was allowed - this should have failed!")
    except ValueError as e:
        if "NO IMMEDIATE REWARDS ALLOWED" not in str(e):
            raise AssertionError(f"Wrong error message: {e}")
    
    print("✅ VERIFIED: No immediate rewards allowed - delayed attribution only")


if __name__ == "__main__":
    # Run verification
    verify_no_immediate_rewards()
    
    # Demo usage
    tracker = UserJourneyTracker()
    
    # Add some touchpoints (no immediate rewards)
    user_id = "demo_user_123"
    
    # First touchpoint - search click
    tp1_id = tracker.add_touchpoint(
        user_id=user_id,
        channel="search",
        touchpoint_type=TouchpointType.CLICK,
        campaign_id="search_campaign_001",
        creative_id="search_ad_v1",
        placement_id="google_search_top",
        bid_amount=2.5,
        cost=1.8,
        state_data={"page": "search_results", "query": "parental control app"},
        action_data={"click_position": 1, "ad_type": "search"}
    )
    
    # Second touchpoint - display impression (3 days later)
    import time
    time.sleep(0.1)  # Small delay for demo
    
    tp2_id = tracker.add_touchpoint(
        user_id=user_id,
        channel="display",
        touchpoint_type=TouchpointType.IMPRESSION,
        campaign_id="display_retargeting_001",
        creative_id="banner_ad_v2",
        placement_id="family_blog_sidebar",
        bid_amount=0.8,
        cost=0.05,
        state_data={"page": "blog_article", "content_topic": "screen_time"},
        action_data={"impression_duration": 1200, "ad_format": "banner"}
    )
    
    # Third touchpoint - social engagement (1 week later)
    tp3_id = tracker.add_touchpoint(
        user_id=user_id,
        channel="social",
        touchpoint_type=TouchpointType.ENGAGEMENT,
        campaign_id="facebook_video_001",
        creative_id="video_testimonial_v1",
        placement_id="facebook_feed",
        bid_amount=1.2,
        cost=0.9,
        state_data={"platform": "facebook", "feed_position": 3},
        action_data={"engagement_type": "video_view", "view_duration": 45}
    )
    
    print(f"Added 3 touchpoints for user {user_id} (all with 0.0 immediate reward)")
    
    # Record conversion (triggers delayed attribution)
    delayed_rewards = tracker.record_conversion(
        user_id=user_id,
        conversion_type=ConversionType.TRIAL,
        value=120.0,  # $120 subscription value
        metadata={"trial_length_days": 14}
    )
    
    print(f"Conversion recorded - generated {len(delayed_rewards)} delayed reward attributions")
    
    # Show attribution results
    for reward in delayed_rewards:
        print(f"\nAttribution Model: {reward.attribution_model}")
        print(f"Attribution Window: {reward.attribution_window_days} days")
        print(f"Total Value: ${reward.total_attributed_value:.2f}")
        print("Touchpoint Credits:")
        for touchpoint, credit in reward.attributed_touchpoints:
            print(f"  - {touchpoint.touchpoint_id[:8]}... ({touchpoint.channel}): ${credit:.2f}")
    
    # Check attributed rewards for specific touchpoints
    print(f"\nDelayed rewards for touchpoint {tp1_id}:")
    tp1_rewards = tracker.get_attributed_rewards_for_touchpoint(tp1_id)
    for reward_data in tp1_rewards:
        print(f"  - Model: {reward_data['attribution_model']}, "
              f"Credit: ${reward_data['attributed_value']:.2f}, "
              f"Time to conversion: {reward_data['time_to_conversion_hours']:.1f}h")
    
    # Get statistics
    stats = tracker.get_journey_statistics()
    print(f"\nJourney Statistics:")
    print(f"  - Total users: {stats['total_users']}")
    print(f"  - Total touchpoints: {stats['total_touchpoints']}")
    print(f"  - Total conversions: {stats['total_conversions']}")
    print(f"  - Total delayed rewards: {stats['total_delayed_rewards']}")
    print(f"  - Total attributed value: ${stats['total_attributed_value']:.2f}")
    print(f"  - Conversion rate: {stats['conversion_rate']:.1%}")