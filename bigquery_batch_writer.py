#!/usr/bin/env python3
"""
BigQuery Batch Writer for GAELP
Handles batch writes to avoid quota exceeded errors
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

logger = logging.getLogger(__name__)

class BigQueryBatchWriter:
    """
    Batch writer for BigQuery to avoid DML quota limits.
    Accumulates writes and executes them in batches.
    """
    
    def __init__(self, 
                 project_id: str,
                 dataset_id: str,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_retries: int = 3):
        """
        Initialize batch writer.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            batch_size: Number of records to batch before writing
            flush_interval: Time in seconds between automatic flushes
            max_retries: Maximum retry attempts for failed writes
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        
        # Initialize BigQuery client
        self.client = bigquery.Client(project=project_id)
        
        # Batch queues for different tables
        self.batches = {
            'persistent_users': deque(),
            'journey_sessions': deque(),
            'persistent_touchpoints': deque(),
            'competitor_exposures': deque()
        }
        
        # Thread safety
        self.lock = threading.Lock()
        self.shutdown = False
        
        # Statistics
        self.stats = {
            'total_writes': 0,
            'successful_writes': 0,
            'failed_writes': 0,
            'batch_flushes': 0,
            'quota_errors': 0
        }
        
        # Start background flush thread
        self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self.flush_thread.start()
        
        logger.info(f"BigQueryBatchWriter initialized with batch_size={batch_size}, "
                   f"flush_interval={flush_interval}s")
    
    def add_user_update(self, user_data: Dict[str, Any]):
        """Add user update to batch queue."""
        with self.lock:
            self.batches['persistent_users'].append({
                'data': user_data,
                'timestamp': datetime.now(),
                'operation': 'update'
            })
            
            # Check if we should flush
            if len(self.batches['persistent_users']) >= self.batch_size:
                self._flush_table('persistent_users')
    
    def add_user_insert(self, user_data: Dict[str, Any]):
        """Add user insert to batch queue."""
        with self.lock:
            self.batches['persistent_users'].append({
                'data': user_data,
                'timestamp': datetime.now(),
                'operation': 'insert'
            })
            
            if len(self.batches['persistent_users']) >= self.batch_size:
                self._flush_table('persistent_users')
    
    def add_session(self, session_data: Dict[str, Any]):
        """Add session to batch queue."""
        with self.lock:
            self.batches['journey_sessions'].append({
                'data': session_data,
                'timestamp': datetime.now(),
                'operation': 'insert'
            })
            
            if len(self.batches['journey_sessions']) >= self.batch_size:
                self._flush_table('journey_sessions')
    
    def add_touchpoint(self, touchpoint_data: Dict[str, Any]):
        """Add touchpoint to batch queue."""
        with self.lock:
            self.batches['persistent_touchpoints'].append({
                'data': touchpoint_data,
                'timestamp': datetime.now(),
                'operation': 'insert'
            })
            
            if len(self.batches['persistent_touchpoints']) >= self.batch_size:
                self._flush_table('persistent_touchpoints')
    
    def _flush_table(self, table_name: str):
        """Flush a specific table's batch queue."""
        if not self.batches[table_name]:
            return
        
        batch = list(self.batches[table_name])
        self.batches[table_name].clear()
        
        # Separate by operation type
        inserts = [item for item in batch if item['operation'] == 'insert']
        updates = [item for item in batch if item['operation'] == 'update']
        
        # Execute batch operations
        if inserts:
            self._execute_batch_insert(table_name, inserts)
        
        if updates:
            self._execute_batch_update(table_name, updates)
        
        self.stats['batch_flushes'] += 1
    
    def _execute_batch_insert(self, table_name: str, items: List[Dict]):
        """Execute batch insert using streaming API."""
        table_ref = self.client.dataset(self.dataset_id).table(table_name)
        
        # Prepare rows for insertion
        rows_to_insert = []
        for item in items:
            row_data = item['data'].copy()
            # Add metadata
            if 'created_at' not in row_data:
                row_data['created_at'] = item['timestamp'].isoformat()
            rows_to_insert.append(row_data)
        
        # Retry logic for quota errors
        for attempt in range(self.max_retries):
            try:
                errors = self.client.insert_rows_json(table_ref, rows_to_insert)
                
                if errors:
                    logger.error(f"Batch insert errors for {table_name}: {errors}")
                    self.stats['failed_writes'] += len(items)
                else:
                    self.stats['successful_writes'] += len(items)
                    logger.debug(f"Successfully inserted {len(items)} rows to {table_name}")
                
                break
                
            except GoogleCloudError as e:
                if "quota" in str(e).lower():
                    self.stats['quota_errors'] += 1
                    wait_time = min(2 ** attempt, 30)  # Exponential backoff
                    logger.warning(f"Quota error on {table_name}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to insert batch to {table_name}: {e}")
                    self.stats['failed_writes'] += len(items)
                    break
        
        self.stats['total_writes'] += len(items)
    
    def _execute_batch_update(self, table_name: str, items: List[Dict]):
        """Execute batch update using MERGE statement."""
        if not items:
            return
        
        # Create temporary table for staging
        temp_table_id = f"{table_name}_temp_{int(time.time())}"
        temp_table_ref = self.client.dataset(self.dataset_id).table(temp_table_id)
        
        try:
            # Create temporary table with same schema
            source_table = self.client.get_table(
                self.client.dataset(self.dataset_id).table(table_name)
            )
            temp_table = bigquery.Table(temp_table_ref, schema=source_table.schema)
            temp_table = self.client.create_table(temp_table)
            
            # Insert data into temp table
            rows_to_insert = [item['data'] for item in items]
            errors = self.client.insert_rows_json(temp_table_ref, rows_to_insert)
            
            if errors:
                logger.error(f"Failed to insert into temp table: {errors}")
                self.stats['failed_writes'] += len(items)
                return
            
            # Execute MERGE statement
            merge_query = self._build_merge_query(table_name, temp_table_id)
            
            job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
            job = self.client.query(merge_query, job_config=job_config)
            job.result()  # Wait for completion
            
            self.stats['successful_writes'] += len(items)
            logger.debug(f"Successfully merged {len(items)} updates to {table_name}")
            
        except Exception as e:
            logger.error(f"Batch update failed for {table_name}: {e}")
            self.stats['failed_writes'] += len(items)
            
            # Check if quota error
            if "quota" in str(e).lower():
                self.stats['quota_errors'] += 1
                # Re-queue items for retry
                with self.lock:
                    self.batches[table_name].extend(items)
        
        finally:
            # Clean up temp table
            try:
                self.client.delete_table(temp_table_ref, not_found_ok=True)
            except:
                pass
            
            self.stats['total_writes'] += len(items)
    
    def _build_merge_query(self, table_name: str, temp_table_id: str) -> str:
        """Build MERGE query based on table structure."""
        
        if table_name == 'persistent_users':
            return f"""
            MERGE `{self.project_id}.{self.dataset_id}.{table_name}` T
            USING `{self.project_id}.{self.dataset_id}.{temp_table_id}` S
            ON T.canonical_user_id = S.canonical_user_id
            WHEN MATCHED THEN
                UPDATE SET
                    device_ids = S.device_ids,
                    current_journey_state = S.current_journey_state,
                    awareness_level = S.awareness_level,
                    fatigue_score = S.fatigue_score,
                    intent_score = S.intent_score,
                    competitor_exposures = S.competitor_exposures,
                    competitor_fatigue = S.competitor_fatigue,
                    devices_seen = S.devices_seen,
                    cross_device_confidence = S.cross_device_confidence,
                    last_seen = S.last_seen,
                    last_episode = S.last_episode,
                    episode_count = S.episode_count,
                    journey_history = S.journey_history,
                    touchpoint_history = S.touchpoint_history,
                    conversion_history = S.conversion_history,
                    timeout_at = S.timeout_at,
                    is_active = S.is_active,
                    updated_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT ROW
            """
        
        elif table_name == 'journey_sessions':
            return f"""
            MERGE `{self.project_id}.{self.dataset_id}.{table_name}` T
            USING `{self.project_id}.{self.dataset_id}.{temp_table_id}` S
            ON T.session_id = S.session_id
            WHEN MATCHED THEN
                UPDATE SET
                    session_end = S.session_end,
                    session_state_changes = S.session_state_changes,
                    session_touchpoints = S.session_touchpoints,
                    session_channels = S.session_channels,
                    session_devices = S.session_devices,
                    converted_in_session = S.converted_in_session,
                    conversion_value = S.conversion_value,
                    session_engagement = S.session_engagement
            WHEN NOT MATCHED THEN
                INSERT ROW
            """
        
        else:
            # Generic merge for other tables
            return f"""
            MERGE `{self.project_id}.{self.dataset_id}.{table_name}` T
            USING `{self.project_id}.{self.dataset_id}.{temp_table_id}` S
            ON T.id = S.id
            WHEN NOT MATCHED THEN
                INSERT ROW
            """
    
    def _background_flush(self):
        """Background thread to periodically flush batches."""
        while not self.shutdown:
            time.sleep(self.flush_interval)
            
            with self.lock:
                for table_name in self.batches:
                    if self.batches[table_name]:
                        self._flush_table(table_name)
    
    def flush_all(self):
        """Manually flush all pending batches."""
        with self.lock:
            for table_name in self.batches:
                if self.batches[table_name]:
                    self._flush_table(table_name)
        
        logger.info("All batches flushed")
    
    def get_stats(self) -> Dict[str, int]:
        """Get writer statistics."""
        return self.stats.copy()
    
    def shutdown_writer(self):
        """Shutdown the writer and flush remaining data."""
        logger.info("Shutting down BigQueryBatchWriter...")
        self.shutdown = True
        
        # Final flush
        self.flush_all()
        
        # Wait for flush thread
        self.flush_thread.join(timeout=10)
        
        logger.info(f"BigQueryBatchWriter shutdown complete. Stats: {self.stats}")