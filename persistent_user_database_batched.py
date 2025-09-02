"""
GAELP Persistent User Database with Batch Writing
Modified to use batch writer to avoid BigQuery quota issues
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from persistent_user_database import PersistentUserDatabase, PersistentUser
from bigquery_batch_writer import BigQueryBatchWriter

logger = logging.getLogger(__name__)

class BatchedPersistentUserDatabase(PersistentUserDatabase):
    """
    Extended version of PersistentUserDatabase that uses batch writing
    to avoid BigQuery quota exceeded errors.
    """
    
    def __init__(self,
                 project_id: str = None,
                 dataset_id: str = "gaelp_users", 
                 timeout_days: int = 14,
                 use_batch_writer: bool = True,
                 batch_size: int = 100,
                 flush_interval: float = 5.0):
        """
        Initialize with optional batch writer.
        
        Args:
            use_batch_writer: Whether to use batch writing (recommended)
            batch_size: Number of operations to batch before writing
            flush_interval: Seconds between automatic flushes
        """
        # Initialize parent class
        super().__init__(project_id, dataset_id, timeout_days)
        
        # Initialize batch writer if requested
        self.use_batch_writer = use_batch_writer
        self.batch_writer = None
        
        if use_batch_writer:
            try:
                self.batch_writer = BigQueryBatchWriter(
                    project_id=self.project_id,
                    dataset_id=self.dataset_id,
                    batch_size=batch_size,
                    flush_interval=flush_interval
                )
                logger.info(f"Batch writer initialized with batch_size={batch_size}, "
                           f"flush_interval={flush_interval}s")
            except Exception as e:
                logger.warning(f"Failed to initialize batch writer: {e}. "
                              "Falling back to direct writes.")
                self.use_batch_writer = False
                self.batch_writer = None
    
    def _update_user_in_database(self, user: PersistentUser):
        """Override to use batch writer if available."""
        
        if self.use_batch_writer and self.batch_writer:
            # Prepare user data for batch update
            user_data = {
                'canonical_user_id': user.canonical_user_id,
                'device_ids': list(user.device_ids),
                'current_journey_state': user.current_journey_state,
                'awareness_level': user.awareness_level,
                'fatigue_score': user.fatigue_score,
                'intent_score': user.intent_score,
                'competitor_exposures': json.dumps(user.competitor_exposures) if user.competitor_exposures else '{}',
                'competitor_fatigue': json.dumps(user.competitor_fatigue) if user.competitor_fatigue else '{}',
                'devices_seen': json.dumps({k: v.isoformat() if isinstance(v, datetime) else v 
                                for k, v in user.devices_seen.items()}) if user.devices_seen else '{}',
                'cross_device_confidence': user.cross_device_confidence,
                'last_seen': user.last_seen.isoformat(),
                'last_episode': user.last_episode,
                'episode_count': user.episode_count,
                'journey_history': json.dumps(user.journey_history) if user.journey_history else '[]',
                'touchpoint_history': json.dumps(user.touchpoint_history) if user.touchpoint_history else '[]',
                'conversion_history': json.dumps(user.conversion_history) if user.conversion_history else '[]',
                'timeout_at': user.timeout_at.isoformat(),
                'is_active': user.is_active,
                'updated_at': datetime.now().isoformat()
            }
            
            # Add to batch queue
            self.batch_writer.add_user_update(user_data)
            logger.debug(f"User {user.canonical_user_id} added to batch update queue")
            
        else:
            # Fall back to parent implementation
            super()._update_user_in_database(user)
    
    def _insert_user_into_database(self, user: PersistentUser):
        """Override to use batch writer if available."""
        
        if self.use_batch_writer and self.batch_writer:
            # Prepare user data for batch insert
            user_data = {
                'user_id': user.user_id,
                'canonical_user_id': user.canonical_user_id,
                'device_ids': list(user.device_ids),
                'email_hash': user.email_hash,
                'phone_hash': user.phone_hash,
                'current_journey_state': user.current_journey_state,
                'awareness_level': user.awareness_level,
                'fatigue_score': user.fatigue_score,
                'intent_score': user.intent_score,
                'competitor_exposures': json.dumps(user.competitor_exposures) if user.competitor_exposures else '{}',
                'competitor_fatigue': json.dumps(user.competitor_fatigue) if user.competitor_fatigue else '{}',
                'devices_seen': json.dumps({k: v.isoformat() if isinstance(v, datetime) else v 
                                for k, v in user.devices_seen.items()}) if user.devices_seen else '{}',
                'cross_device_confidence': user.cross_device_confidence,
                'first_seen': user.first_seen.isoformat(),
                'last_seen': user.last_seen.isoformat(),
                'last_episode': user.last_episode,
                'episode_count': user.episode_count,
                'journey_history': json.dumps(user.journey_history) if user.journey_history else '[]',
                'touchpoint_history': json.dumps(user.touchpoint_history) if user.touchpoint_history else '[]',
                'conversion_history': json.dumps(user.conversion_history) if user.conversion_history else '[]',
                'timeout_at': user.timeout_at.isoformat(),
                'is_active': user.is_active,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Add to batch queue
            self.batch_writer.add_user_insert(user_data)
            logger.info(f"User {user.canonical_user_id} added to batch insert queue")
            
        else:
            # Fall back to parent implementation
            super()._insert_user_into_database(user)
    
    def flush_batches(self):
        """Manually flush all pending batches."""
        if self.batch_writer:
            self.batch_writer.flush_all()
            logger.info("All batches flushed to BigQuery")
    
    def get_batch_stats(self) -> Optional[Dict[str, int]]:
        """Get batch writer statistics."""
        if self.batch_writer:
            return self.batch_writer.get_stats()
        return None
    
    def shutdown(self):
        """Shutdown the database and flush remaining batches."""
        if self.batch_writer:
            self.batch_writer.shutdown_writer()
            logger.info("Batch writer shutdown complete")
    
    def __del__(self):
        """Ensure batches are flushed on deletion."""
        try:
            self.shutdown()
        except:
            pass