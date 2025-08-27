"""
GAELP Persistent User Integration - CRITICAL SYSTEM BRIDGE

This module integrates the persistent user database with the existing GAELP system,
ensuring users maintain state across episodes and RL training is valid.

CRITICAL FIXES:
- Users NEVER reset between episodes
- State persists across training runs
- Cross-device identity resolution
- Proper reward attribution over time
"""

import os
import sys
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, '/home/hariravichandran/AELP')

from persistent_user_database import (
    PersistentUserDatabase,
    PersistentUser,
    JourneySession,
    TouchpointRecord,
    CompetitorExposureRecord
)

# Import existing GAELP components
try:
    from user_journey_database import UserJourneyDatabase, JourneyState
    from journey_state import JourneyState as JourneyStateEnum, JourneyStateManager
except ImportError as e:
    logging.warning(f"Could not import existing journey components: {e}")
    # Create minimal versions for compatibility
    class JourneyState:
        UNAWARE = "UNAWARE"
        AWARE = "AWARE"
        CONSIDERING = "CONSIDERING"
        INTENT = "INTENT"
        CONVERTED = "CONVERTED"

logger = logging.getLogger(__name__)

class GAELPPersistentIntegration:
    """
    Integration layer between persistent user database and GAELP system.
    
    This is the CRITICAL component that ensures users never reset between episodes.
    """
    
    def __init__(self, 
                 project_id: str = None,
                 dataset_id: str = "gaelp_users",
                 timeout_days: int = 14):
        
        # Initialize persistent database - CRITICAL
        self.persistent_db = PersistentUserDatabase(
            project_id=project_id,
            dataset_id=dataset_id,
            timeout_days=timeout_days
        )
        
        # Initialize existing journey database for compatibility
        try:
            self.legacy_db = UserJourneyDatabase(
                project_id=project_id,
                dataset_id="gaelp_data",  # Legacy dataset
                timeout_days=timeout_days
            )
            self.has_legacy_db = True
        except Exception as e:
            logger.warning(f"Legacy database not available: {e}")
            self.has_legacy_db = False
        
        # Episode tracking
        self.current_episode_id = None
        self.active_sessions: Dict[str, JourneySession] = {}
        
        logger.info("GAELP Persistent Integration initialized - Users will NEVER reset!")
    
    def start_episode(self, episode_id: str) -> str:
        """
        Start a new episode - users maintain state from previous episodes.
        
        This is CRITICAL - users do NOT reset between episodes.
        """
        self.current_episode_id = episode_id
        self.active_sessions.clear()
        
        logger.info(f"Starting episode {episode_id} with persistent users")
        
        # Clean up any expired users
        expired_count = self.persistent_db.cleanup_expired_users()
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired users")
        
        return episode_id
    
    def get_or_create_user(self, 
                          user_id: str,
                          device_fingerprint: Dict[str, Any] = None,
                          user_attributes: Dict[str, Any] = None) -> Tuple[PersistentUser, bool]:
        """
        Get or create persistent user - NEVER resets between episodes.
        
        This replaces the legacy user creation that reset between episodes.
        """
        if not self.current_episode_id:
            raise ValueError("Must call start_episode() before getting users")
        
        # Get persistent user - maintains state across episodes
        user, is_new = self.persistent_db.get_or_create_persistent_user(
            user_id=user_id,
            episode_id=self.current_episode_id,
            device_fingerprint=device_fingerprint
        )
        
        if is_new:
            logger.info(f"Created new persistent user {user.canonical_user_id}")
        else:
            logger.info(f"Retrieved persistent user {user.canonical_user_id} - episode #{user.episode_count}, state: {user.current_journey_state}")
        
        return user, is_new
    
    def start_user_journey(self, user: PersistentUser) -> JourneySession:
        """Start a journey session for a persistent user."""
        
        session = self.persistent_db.start_journey_session(
            user=user,
            episode_id=self.current_episode_id
        )
        
        # Track active session
        self.active_sessions[user.canonical_user_id] = session
        
        logger.info(f"Started journey session {session.session_id} for persistent user {user.canonical_user_id}")
        return session
    
    def record_user_interaction(self,
                               user: PersistentUser,
                               channel: str,
                               interaction_type: str = "impression",
                               **interaction_data) -> Tuple[TouchpointRecord, str, str]:
        """
        Record user interaction with persistent state updates.
        
        Returns:
            Tuple of (touchpoint_record, pre_state, post_state)
        """
        # Get or create session
        session = self.active_sessions.get(user.canonical_user_id)
        if not session:
            session = self.start_user_journey(user)
        
        # Record touchpoint with state transition
        pre_state = user.current_journey_state
        
        touchpoint = self.persistent_db.record_touchpoint(
            user=user,
            session=session,
            channel=channel,
            interaction_data={
                'interaction_type': interaction_type,
                **interaction_data
            }
        )
        
        post_state = user.current_journey_state
        
        logger.info(f"User {user.canonical_user_id} interaction: {pre_state} -> {post_state} via {channel}")
        
        return touchpoint, pre_state, post_state
    
    def record_competitor_interaction(self,
                                    user: PersistentUser,
                                    competitor_name: str,
                                    competitor_channel: str,
                                    exposure_type: str = "impression") -> CompetitorExposureRecord:
        """Record competitor exposure with persistent fatigue tracking."""
        
        session = self.active_sessions.get(user.canonical_user_id)
        if not session:
            session = self.start_user_journey(user)
        
        exposure = self.persistent_db.record_competitor_exposure(
            user=user,
            session=session,
            competitor_name=competitor_name,
            competitor_channel=competitor_channel,
            exposure_type=exposure_type
        )
        
        logger.info(f"Competitor exposure for user {user.canonical_user_id}: {competitor_name} via {competitor_channel}")
        
        return exposure
    
    def record_conversion(self,
                         user: PersistentUser,
                         conversion_value: float,
                         conversion_type: str = "purchase") -> None:
        """Record conversion for persistent user."""
        
        session = self.active_sessions.get(user.canonical_user_id)
        if not session:
            session = self.start_user_journey(user)
        
        self.persistent_db.record_conversion(
            user=user,
            session=session,
            conversion_value=conversion_value,
            conversion_type=conversion_type
        )
        
        logger.info(f"Conversion recorded for user {user.canonical_user_id}: ${conversion_value}")
    
    def get_user_state_for_rl(self, user: PersistentUser) -> Dict[str, Any]:
        """
        Get user state vector for RL agent - includes persistent history.
        
        This is CRITICAL for proper RL learning - state includes full user history.
        """
        
        # Get comprehensive user analytics
        analytics = self.persistent_db.get_user_analytics(user.canonical_user_id)
        
        # Create state vector for RL agent
        state_vector = {
            # Current state
            'current_journey_state': user.current_journey_state,
            'awareness_level': user.awareness_level,
            'fatigue_score': user.fatigue_score,
            'intent_score': user.intent_score,
            
            # Historical context (CRITICAL for RL)
            'episode_count': user.episode_count,
            'total_touchpoints': len(user.touchpoint_history),
            'total_conversions': len(user.conversion_history),
            'days_active': analytics.get('days_active', 0),
            
            # Cross-device context
            'device_count': len(user.device_ids),
            'cross_device_confidence': user.cross_device_confidence,
            
            # Competitor context
            'competitor_fatigue': user.competitor_fatigue,
            'total_competitor_exposures': sum(len(exposures) for exposures in user.competitor_exposures.values()),
            
            # Time context
            'hours_since_last_seen': (datetime.now() - user.last_seen).total_seconds() / 3600,
            'days_until_timeout': (user.timeout_at - datetime.now()).days if user.timeout_at else 14,
            
            # Engagement patterns
            'avg_engagement': analytics.get('avg_engagement', 0.0),
            'channels_used_count': len(analytics.get('channels_used', [])),
            
            # State progression
            'state_changes_count': len(analytics.get('state_changes', [])),
            'last_state_change_hours': self._hours_since_last_state_change(user),
            
            # Conversion history
            'conversion_rate': len(user.conversion_history) / max(1, user.episode_count),
            'avg_conversion_value': self._avg_conversion_value(user),
            'days_since_last_conversion': self._days_since_last_conversion(user),
        }
        
        return state_vector
    
    def calculate_reward_for_rl(self, 
                               user: PersistentUser,
                               pre_state: str,
                               post_state: str,
                               action_taken: str,
                               touchpoint: TouchpointRecord) -> float:
        """
        Calculate reward for RL agent based on persistent user state.
        
        This uses the full user history for proper reward calculation.
        """
        
        base_reward = 0.0
        
        # State progression rewards (higher for later states)
        state_values = {
            "UNAWARE": 0.0,
            "AWARE": 0.2, 
            "CONSIDERING": 0.5,
            "INTENT": 0.8,
            "CONVERTED": 1.0
        }
        
        # Reward for positive state progression
        if post_state != pre_state:
            pre_value = state_values.get(pre_state, 0.0)
            post_value = state_values.get(post_state, 0.0)
            
            if post_value > pre_value:
                base_reward += (post_value - pre_value) * 10.0
            else:
                base_reward -= 2.0  # Penalty for regression
        
        # Engagement reward
        if touchpoint.engagement_score > 0:
            base_reward += touchpoint.engagement_score * 5.0
        
        # Conversion reward (major reward)
        if post_state == "CONVERTED":
            base_reward += 100.0
            
            # Bonus for efficient conversion (fewer episodes)
            if user.episode_count <= 3:
                base_reward += 50.0
        
        # Fatigue penalty (accumulated over time)
        fatigue_penalty = user.fatigue_score * 10.0
        base_reward -= fatigue_penalty
        
        # Competitor impact penalty
        total_competitor_fatigue = sum(user.competitor_fatigue.values())
        base_reward -= total_competitor_fatigue * 5.0
        
        # Long-term user penalty (encourage retention)
        if user.episode_count > 10 and post_state in ["UNAWARE", "AWARE"]:
            base_reward -= 5.0  # Penalty for not progressing long-term users
        
        # Efficiency bonus (reward for progressing users with fewer touchpoints)
        touchpoints_in_episode = sum(1 for tp in user.touchpoint_history 
                                   if tp.get('episode_id') == self.current_episode_id)
        if touchpoints_in_episode > 0 and post_state != pre_state:
            efficiency_bonus = max(0, 5.0 - touchpoints_in_episode)
            base_reward += efficiency_bonus
        
        logger.debug(f"Reward calculation for user {user.canonical_user_id}: "
                    f"base={base_reward:.2f}, fatigue_penalty={fatigue_penalty:.2f}")
        
        return base_reward
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode with persistent user stats."""
        
        if not self.current_episode_id:
            return {}
        
        # Count users by state
        state_distribution = {}
        total_users = 0
        total_sessions = len(self.active_sessions)
        total_conversions = 0
        total_new_users = 0
        
        # Get basic stats from active sessions
        for session in self.active_sessions.values():
            total_users += 1
            if session.converted_in_session:
                total_conversions += 1
        
        # Get more detailed analytics from persistent DB
        try:
            # This would be a custom query to get episode-specific stats
            # For now, return basic information
            pass
        except Exception as e:
            logger.warning(f"Could not get detailed episode stats: {e}")
        
        return {
            'episode_id': self.current_episode_id,
            'total_active_sessions': total_sessions,
            'total_conversions': total_conversions,
            'conversion_rate': total_conversions / max(1, total_sessions),
            'state_distribution': state_distribution,
            'persistent_users_active': len(self.active_sessions)
        }
    
    def end_episode(self) -> Dict[str, Any]:
        """End current episode and return summary."""
        
        summary = self.get_episode_summary()
        
        # End all active sessions
        for session in self.active_sessions.values():
            if not session.session_end:
                session.session_end = datetime.now()
                # Update in database
                try:
                    self.persistent_db._update_session_in_database(session)
                except Exception as e:
                    logger.error(f"Failed to end session {session.session_id}: {e}")
        
        # Clear active sessions
        self.active_sessions.clear()
        
        logger.info(f"Episode {self.current_episode_id} ended - "
                   f"{summary.get('total_active_sessions', 0)} sessions, "
                   f"{summary.get('total_conversions', 0)} conversions")
        
        self.current_episode_id = None
        return summary
    
    # Helper methods
    
    def _hours_since_last_state_change(self, user: PersistentUser) -> float:
        """Calculate hours since last state change."""
        
        if not user.touchpoint_history:
            return 999.0  # Very high value if no history
        
        # Find last state change
        for touchpoint_data in reversed(user.touchpoint_history):
            if touchpoint_data.get('pre_state') != touchpoint_data.get('post_state'):
                last_change = datetime.fromisoformat(touchpoint_data['timestamp'].replace('Z', '+00:00'))
                return (datetime.now() - last_change).total_seconds() / 3600
        
        return 999.0  # No state changes found
    
    def _avg_conversion_value(self, user: PersistentUser) -> float:
        """Calculate average conversion value."""
        
        if not user.conversion_history:
            return 0.0
        
        values = [conv.get('conversion_value', 0.0) for conv in user.conversion_history]
        return sum(values) / len(values)
    
    def _days_since_last_conversion(self, user: PersistentUser) -> float:
        """Calculate days since last conversion."""
        
        if not user.conversion_history:
            return 999.0  # Very high value if no conversions
        
        last_conversion = user.conversion_history[-1]
        last_conv_time = datetime.fromisoformat(last_conversion['timestamp'].replace('Z', '+00:00'))
        return (datetime.now() - last_conv_time).total_seconds() / (24 * 3600)
    
    def migrate_legacy_users(self) -> int:
        """
        Migrate users from legacy system to persistent system.
        
        This is for transitioning from the old user reset system.
        """
        if not self.has_legacy_db:
            logger.info("No legacy database to migrate from")
            return 0
        
        migrated_count = 0
        
        try:
            # Get active journeys from legacy system
            # Implementation would depend on legacy system structure
            logger.info("Legacy user migration not yet implemented")
            # TODO: Implement migration logic
            
        except Exception as e:
            logger.error(f"Failed to migrate legacy users: {e}")
        
        return migrated_count


# Integration testing function
def test_persistent_integration():
    """Test the persistent integration system."""
    
    print("=== Testing GAELP Persistent Integration ===")
    
    try:
        # Initialize integration
        integration = GAELPPersistentIntegration()
        
        # Start episode
        episode_id = f"test_episode_{uuid.uuid4().hex[:8]}"
        integration.start_episode(episode_id)
        print(f"Started episode: {episode_id}")
        
        # Create user
        user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        user, is_new = integration.get_or_create_user(user_id)
        print(f"User created: {user.canonical_user_id}, new: {is_new}")
        
        # Record interactions
        touchpoint1, pre1, post1 = integration.record_user_interaction(
            user, "google_ads", "impression", engagement_score=0.7
        )
        print(f"Interaction 1: {pre1} -> {post1}")
        
        touchpoint2, pre2, post2 = integration.record_user_interaction(
            user, "facebook_ads", "click", engagement_score=0.9, dwell_time=45.0
        )
        print(f"Interaction 2: {pre2} -> {post2}")
        
        # Record competitor exposure
        competitor_exp = integration.record_competitor_interaction(
            user, "competitor_A", "display_ads"
        )
        print(f"Competitor exposure: {competitor_exp.competitor_name}")
        
        # Get RL state
        rl_state = integration.get_user_state_for_rl(user)
        print(f"RL State: {rl_state['current_journey_state']}, "
              f"Episodes: {rl_state['episode_count']}, "
              f"Fatigue: {rl_state['fatigue_score']:.3f}")
        
        # Calculate reward
        reward = integration.calculate_reward_for_rl(user, pre2, post2, "show_ad", touchpoint2)
        print(f"RL Reward: {reward:.2f}")
        
        # End episode
        summary = integration.end_episode()
        print(f"Episode ended: {summary}")
        
        # START NEW EPISODE - USER SHOULD PERSIST
        episode_id_2 = f"test_episode_{uuid.uuid4().hex[:8]}"
        integration.start_episode(episode_id_2)
        
        # Same user should maintain state
        user2, is_new2 = integration.get_or_create_user(user_id)
        print(f"Episode 2 - User: {user2.canonical_user_id}, new: {is_new2}")
        print(f"State persisted: {user2.current_journey_state}, Episodes: {user2.episode_count}")
        
        if user2.episode_count > 1:
            print("✅ USER STATE PERSISTED ACROSS EPISODES!")
        else:
            print("❌ User state did not persist")
            
        print("\n✅ GAELP Persistent Integration Test Complete")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_persistent_integration()