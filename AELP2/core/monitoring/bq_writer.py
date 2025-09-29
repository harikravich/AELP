"""
Production-grade BigQuery monitoring and telemetry writer for AELP2.

This module provides robust, fault-tolerant BigQuery integration for training
telemetry, safety events, and A/B test results with proper error handling,
retry logic, and batch writes.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass
from queue import Queue, Empty
import threading
import uuid

try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound, BadRequest, GoogleCloudError
    from google.api_core import retry
    BIGQUERY_AVAILABLE = True
except ImportError as e:
    bigquery = None
    NotFound = BadRequest = GoogleCloudError = retry = None
    BIGQUERY_AVAILABLE = False
    IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


@dataclass
class BigQueryConfig:
    """Configuration for BigQuery writer."""
    project_id: str
    training_dataset: str
    users_dataset: Optional[str] = None
    batch_size: int = 100
    flush_interval: int = 30  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    @classmethod
    def from_env(cls) -> 'BigQueryConfig':
        """Create config from environment variables."""
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
        
        training_dataset = os.getenv("BIGQUERY_TRAINING_DATASET")
        if not training_dataset:
            raise ValueError("BIGQUERY_TRAINING_DATASET environment variable is required")
        users_dataset = os.getenv("BIGQUERY_USERS_DATASET")
        
        return cls(
            project_id=project_id,
            training_dataset=training_dataset,
            users_dataset=users_dataset
        )


class BigQueryConnectionError(Exception):
    """Raised when BigQuery connection fails."""
    pass


class BigQueryWriteError(Exception):
    """Raised when BigQuery write operation fails."""
    pass


class BigQueryWriter:
    """Production-grade BigQuery writer with batching, retries, and error handling."""
    
    def __init__(self, config: Optional[BigQueryConfig] = None):
        """Initialize BigQuery writer.
        
        Args:
            config: BigQuery configuration. If None, loads from environment.
            
        Raises:
            BigQueryConnectionError: If BigQuery client cannot be initialized.
        """
        if not BIGQUERY_AVAILABLE:
            raise BigQueryConnectionError(
                f"BigQuery client library not available. Install with: "
                f"pip install google-cloud-bigquery\n"
                f"Import error: {IMPORT_ERROR}"
            )
        
        self.config = config or BigQueryConfig.from_env()
        self._client = None
        self._batch_queue = Queue()
        self._batch_thread = None
        self._shutdown_event = threading.Event()
        
        # Initialize client and validate connection
        self._initialize_client()
        self.validate_connection()
        
        # Start batch processing thread
        self._start_batch_processor()
        
        # Define table schemas
        self._schemas = {
            'training_episodes': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("episode_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("episode_index", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("steps", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("auctions", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("wins", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("spend", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("revenue", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("conversions", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("win_rate", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("avg_cpc", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("epsilon", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("cac", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("roas", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("step_details", "JSON", mode="NULLABLE"),
            ],
            'bidding_events': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("episode_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("step", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("campaign_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("bid_amount", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("won", "BOOL", mode="NULLABLE"),
                bigquery.SchemaField("price_paid", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("auction_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("context", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("explain", "JSON", mode="NULLABLE"),
            ],
            'training_runs': [
                bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("start_time", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("end_time", "TIMESTAMP", mode="NULLABLE"),
                bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("episodes_requested", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("episodes_completed", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("validation_errors", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("training_duration_seconds", "FLOAT", mode="NULLABLE"),
                bigquery.SchemaField("configuration", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("calibration_info", "JSON", mode="NULLABLE"),
            ],
            'safety_events': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("episode_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("severity", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("action_taken", "STRING", mode="NULLABLE"),
            ],
            'ab_experiments': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("experiment_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("variant", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("metrics", "JSON", mode="REQUIRED"),
                bigquery.SchemaField("user_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
            ],
            'subagent_events': [
                bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("subagent", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("episode_id", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            ]
        }
        # Schema compatibility flags (auto-detected)
        self._safety_schema_variant: Optional[str] = None  # 'v2' (expected) | 'legacy' | None
    
    def _initialize_client(self):
        """Initialize BigQuery client with proper error handling."""
        try:
            self._client = bigquery.Client(project=self.config.project_id)
            logger.info(f"BigQuery client initialized for project: {self.config.project_id}")
        except Exception as e:
            raise BigQueryConnectionError(
                f"Failed to initialize BigQuery client: {e}\n"
                f"Ensure GOOGLE_APPLICATION_CREDENTIALS is set or you're running on GCP."
            ) from e
    
    def validate_connection(self):
        """Validate BigQuery connection by attempting to query information schema."""
        if not self._client:
            raise BigQueryConnectionError("BigQuery client not initialized")
        
        try:
            # Test connection with minimal query
            query = f"SELECT 1 as test_connection LIMIT 1"
            query_job = self._client.query(query)
            list(query_job.result())  # Force execution
            logger.info("BigQuery connection validated successfully")
        except Exception as e:
            raise BigQueryConnectionError(
                f"BigQuery connection validation failed: {e}"
            ) from e
    
    def create_tables_if_not_exist(self):
        """Create all required tables if they don't exist."""
        dataset_ref = self._client.dataset(self.config.training_dataset)
        
        # Ensure dataset exists
        try:
            self._client.get_dataset(dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            # Optional dataset location via env var; if not set, let API default
            dataset_location = os.getenv('BIGQUERY_DATASET_LOCATION')
            if dataset_location:
                dataset.location = dataset_location
            self._client.create_dataset(dataset)
            logger.info(f"Created dataset: {self.config.training_dataset}")
        
        # Create or reconcile tables
        partition_field_map = {
            'training_runs': 'start_time',
        }
        for table_name, schema in self._schemas.items():
            table_ref = dataset_ref.table(table_name)
            try:
                table = self._client.get_table(table_ref)
                # Reconcile: add any missing fields as NULLABLE to avoid insert errors
                existing_fields = {f.name: f for f in table.schema}
                missing = [f for f in schema if f.name not in existing_fields]
                if missing:
                    new_schema = list(table.schema)
                    for f in missing:
                        # Add as NULLABLE to remain backward compatible
                        new_schema.append(bigquery.SchemaField(f.name, f.field_type, mode="NULLABLE"))
                    table.schema = new_schema
                    self._client.update_table(table, ["schema"])  # type: ignore
                    logger.info(f"Updated schema for {table_name}: added {', '.join(f.name for f in missing)}")
                else:
                    logger.debug(f"Table {table_name} already exists with required fields")
            except NotFound:
                table = bigquery.Table(table_ref, schema=schema)
                # Choose partition field if present in schema, else create without field partitioning
                desired_field = partition_field_map.get(table_name, 'timestamp')
                if any(f.name == desired_field for f in schema):
                    table.time_partitioning = bigquery.TimePartitioning(
                        type_=bigquery.TimePartitioningType.DAY,
                        field=desired_field
                    )
                else:
                    # Skip field-based partitioning to avoid creation errors
                    logger.info(
                        f"Creating table {table_name} without field partitioning (missing '{desired_field}' in schema)"
                    )
                self._client.create_table(table)
                logger.info(f"Created table: {table_name}")

        # Detect safety_events schema variant for compatibility
        try:
            table = self._client.get_table(dataset_ref.table('safety_events'))
            field_names = {f.name for f in table.schema}
            if {'event_type', 'severity', 'timestamp'}.issubset(field_names):
                self._safety_schema_variant = 'v2'
            elif {'component', 'severity', 'message', 'timestamp'}.issubset(field_names):
                self._safety_schema_variant = 'legacy'
            else:
                self._safety_schema_variant = 'unknown'
                logger.warning(
                    f"Detected unknown safety_events schema: {field_names}. "
                    f"Writer expects v2 fields {['timestamp','episode_id','event_type','severity','metadata','action_taken']}."
                )
        except Exception as e:
            logger.warning(f"Could not detect safety_events schema variant: {e}")
    
    def _start_batch_processor(self):
        """Start background thread for batch processing."""
        if self._batch_thread and self._batch_thread.is_alive():
            return
        
        self._batch_thread = threading.Thread(
            target=self._batch_processor,
            daemon=True
        )
        self._batch_thread.start()
        logger.info("Started batch processor thread")
    
    def _batch_processor(self):
        """Background batch processor that periodically flushes queued writes."""
        batch = []
        last_flush = time.time()
        
        while not self._shutdown_event.is_set():
            try:
                # Check for new items with timeout
                try:
                    item = self._batch_queue.get(timeout=1.0)
                    batch.append(item)
                except Empty:
                    # No items available; continue to check flush conditions
                    pass
                
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_flush >= self.config.flush_interval)
                )
                
                if should_flush and batch:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush = current_time
                    
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                time.sleep(1.0)
        
        # Flush remaining items on shutdown
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, batch: List[Dict[str, Any]]):
        """Flush a batch of writes to BigQuery."""
        if not batch:
            return
        
        # Group by table
        table_batches = {}
        for item in batch:
            table_name = item['_table']
            data = item['_data']
            
            if table_name not in table_batches:
                table_batches[table_name] = []
            table_batches[table_name].append(data)
        
        # Write each table batch
        for table_name, rows in table_batches.items():
            try:
                self._write_rows_with_retry(table_name, rows)
                logger.debug(f"Successfully wrote {len(rows)} rows to {table_name}")
            except Exception as e:
                logger.error(f"Failed to write batch to {table_name}: {e}")
    
    def _write_rows_with_retry(self, table_name: str, rows: List[Dict[str, Any]]):
        """Write rows to BigQuery with exponential backoff retry."""
        table_ref = self._client.dataset(self.config.training_dataset).table(table_name)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                errors = self._client.insert_rows_json(table_ref, rows)
                if errors:
                    raise BigQueryWriteError(f"BigQuery insert errors: {errors}")
                return
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise BigQueryWriteError(f"Failed to write after {self.config.max_retries} retries: {e}")
                
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Write attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)
    
    def write_episode_metrics(self, episode_data: Dict[str, Any]):
        """Write training episode metrics to BigQuery.
        
        Args:
            episode_data: Dictionary containing episode metrics.
                Required fields: episode_id, steps, spend, revenue, conversions, 
                                win_rate, avg_cpc, epsilon, model_version
        """
        # Validate required fields
        required_fields = [
            'episode_id', 'steps', 'spend', 'revenue', 'conversions',
            'win_rate', 'avg_cpc', 'epsilon', 'model_version'
        ]
        
        for field in required_fields:
            if field not in episode_data:
                raise ValueError(f"Required field '{field}' missing from episode_data")
        
        # Add timestamp if not present
        data = dict(episode_data)
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(timezone.utc).isoformat()

        # Sanitize numerics: BigQuery JSON does not accept NaN/Infinity.
        def _sanitize_numeric(val):
            try:
                # Convert numpy scalars to python float
                if hasattr(val, 'item'):
                    val = val.item()
                f = float(val)
            except Exception:
                return None
            if f != f or f in (float('inf'), float('-inf')):  # NaN or Infinity
                return None
            return f

        for key in ['spend', 'revenue', 'win_rate', 'avg_cpc', 'epsilon', 'cac', 'roas']:
            if key in data:
                data[key] = _sanitize_numeric(data[key])

        # Ensure JSON-typed fields are serialized
        if 'step_details' in data and data['step_details'] is not None and not isinstance(data['step_details'], str):
            try:
                data['step_details'] = json.dumps(data['step_details'])
            except Exception:
                data['step_details'] = None

        # Queue for batch processing
        self._batch_queue.put({
            '_table': 'training_episodes',
            '_data': data
        })

    def write_training_run(self, run_data: Dict[str, Any]):
        """Write a training run record (start or end/update) to BigQuery.

        Required fields: run_id, session_id, start_time
        Optional: end_time, status, episodes_requested, episodes_completed, validation_errors,
                  training_duration_seconds, configuration(JSON), calibration_info(JSON)
        """
        required = ['run_id', 'session_id', 'start_time']
        for f in required:
            if f not in run_data:
                raise ValueError(f"Required field '{f}' missing from run_data")

        data = dict(run_data)
        # Coerce times to ISO if provided as datetime
        for tkey in ['start_time', 'end_time']:
            if tkey in data and hasattr(data[tkey], 'isoformat'):
                data[tkey] = data[tkey].isoformat()
        # Ensure JSON serializable fields are serialized
        for jkey in ['configuration', 'calibration_info']:
            if jkey in data and data[jkey] is not None and not isinstance(data[jkey], str):
                try:
                    data[jkey] = json.dumps(data[jkey])
                except Exception:
                    data[jkey] = None

        # Sanitize numeric duration
        if 'training_duration_seconds' in data:
            try:
                val = float(data['training_duration_seconds'])
                if val != val or val in (float('inf'), float('-inf')):
                    data['training_duration_seconds'] = None
                else:
                    data['training_duration_seconds'] = val
            except Exception:
                data['training_duration_seconds'] = None

        self._batch_queue.put({'_table': 'training_runs', '_data': data})
    
    def write_safety_event(self, event_data: Dict[str, Any]):
        """Write safety event to BigQuery.
        
        Args:
            event_data: Dictionary containing safety event data.
                Required fields: event_type, severity
                Optional fields: episode_id, metadata, action_taken
        """
        # Validate required fields
        required_fields = ['event_type', 'severity']
        for field in required_fields:
            if field not in event_data:
                raise ValueError(f"Required field '{field}' missing from event_data")
        
        # Add timestamp if not present
        base = dict(event_data)
        if 'timestamp' not in base:
            base['timestamp'] = datetime.now(timezone.utc).isoformat()

        # Detect schema if not already done
        if self._safety_schema_variant is None:
            try:
                self.create_tables_if_not_exist()  # ensures detection runs
            except Exception:
                pass

        # Map to appropriate schema
        if self._safety_schema_variant == 'legacy':
            # Legacy fields: timestamp, component, severity, message, metadata
            component = base.get('component', 'orchestrator')
            message = base.get('event_type', 'event')
            meta = base.get('metadata') or {}
            # Include original fields in metadata for traceability
            legacy_metadata = {
                'episode_id': base.get('episode_id'),
                'original_event_type': base.get('event_type'),
                'original_metadata': meta,
                'action_taken': base.get('action_taken')
            }
            data = {
                'timestamp': base['timestamp'],
                'component': component,
                'severity': base['severity'],
                'message': message,
                'metadata': json.dumps(legacy_metadata)
            }
        else:
            # v2 (expected) or unknown: attempt v2 write
            data = dict(base)
            if 'metadata' in data and data['metadata'] is not None and not isinstance(data['metadata'], str):
                data['metadata'] = json.dumps(data['metadata'])

        # Queue for batch processing
        self._batch_queue.put({'_table': 'safety_events', '_data': data})
    
    def write_ab_result(self, experiment_data: Dict[str, Any]):
        """Write A/B test result to BigQuery.
        
        Args:
            experiment_data: Dictionary containing A/B test data.
                Required fields: experiment_id, variant, metrics
                Optional fields: user_id, session_id
        """
        # Validate required fields
        required_fields = ['experiment_id', 'variant', 'metrics']
        for field in required_fields:
            if field not in experiment_data:
                raise ValueError(f"Required field '{field}' missing from experiment_data")
        
        # Add timestamp if not present
        data = dict(experiment_data)
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Ensure metrics is JSON serializable
        if not isinstance(data['metrics'], str):
            data['metrics'] = json.dumps(data['metrics'])
        
        # Queue for batch processing
        self._batch_queue.put({
            '_table': 'ab_experiments',
            '_data': data
        })

    def write_bidding_event(self, event_data: Dict[str, Any]):
        """Write a bidding event to BigQuery (env-guarded).

        Guarded by env var `AELP2_BIDDING_EVENTS_ENABLE=1` to prevent writes unless enabled.

        Required fields: bid_amount
        Optional fields: episode_id, step, user_id, campaign_id, won, price_paid, auction_id, context(JSON), explain(JSON)
        """
        if os.getenv('AELP2_BIDDING_EVENTS_ENABLE', '0') != '1':
            return

        if 'bid_amount' not in event_data:
            raise ValueError("Required field 'bid_amount' missing from event_data")

        data = dict(event_data)
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(timezone.utc).isoformat()
        # Ensure JSON fields are serialized if passed as dicts
        for jkey in ['context', 'explain']:
            if jkey in data and data[jkey] is not None and not isinstance(data[jkey], str):
                try:
                    data[jkey] = json.dumps(data[jkey])
                except Exception:
                    data[jkey] = None

        self._batch_queue.put({'_table': 'bidding_events', '_data': data})

    def write_subagent_event(self, subagent_data: Dict[str, Any]):
        """Write a subagent event to BigQuery.

        Args:
            subagent_data: Dict with required keys: subagent, event_type; optional: status, episode_id, metadata
        """
        for field in ['subagent', 'event_type']:
            if field not in subagent_data:
                raise ValueError(f"Required field '{field}' missing from subagent_data")
        data = dict(subagent_data)
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(timezone.utc).isoformat()
        if 'metadata' in data and data['metadata'] is not None and not isinstance(data['metadata'], str):
            data['metadata'] = json.dumps(data['metadata'])
        self._batch_queue.put({'_table': 'subagent_events', '_data': data})
    
    def flush_all(self):
        """Force flush all queued writes immediately."""
        batch = []
        
        # Drain the queue
        while not self._batch_queue.empty():
            try:
                batch.append(self._batch_queue.get_nowait())
            except:
                break
        
        if batch:
            self._flush_batch(batch)
            logger.info(f"Force flushed {len(batch)} queued writes")
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a BigQuery table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information including row count and schema
        """
        table_ref = self._client.dataset(self.config.training_dataset).table(table_name)
        
        try:
            table = self._client.get_table(table_ref)
            return {
                'table_id': table.table_id,
                'num_rows': table.num_rows,
                'num_bytes': table.num_bytes,
                'created': table.created.isoformat() if table.created else None,
                'modified': table.modified.isoformat() if table.modified else None,
                'schema': [
                    {
                        'name': field.name,
                        'field_type': field.field_type,
                        'mode': field.mode
                    }
                    for field in table.schema
                ]
            }
        except NotFound:
            raise ValueError(f"Table {table_name} not found")
    
    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a BigQuery SQL query and return results.
        
        Args:
            sql: SQL query string
            
        Returns:
            List of dictionaries representing query results
        """
        try:
            query_job = self._client.query(sql)
            results = query_job.result()
            
            return [dict(row) for row in results]
        except Exception as e:
            raise BigQueryWriteError(f"Query execution failed: {e}") from e
    
    def close(self):
        """Close the BigQuery writer and flush all pending writes."""
        logger.info("Shutting down BigQuery writer...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Flush remaining writes
        self.flush_all()
        
        # Wait for batch processor to finish
        if self._batch_thread and self._batch_thread.is_alive():
            self._batch_thread.join(timeout=10.0)
        
        logger.info("BigQuery writer shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for easy initialization
def create_bigquery_writer() -> BigQueryWriter:
    """Create a BigQuery writer with environment-based configuration.
    
    Returns:
        Configured BigQueryWriter instance
        
    Raises:
        BigQueryConnectionError: If configuration or connection fails
    """
    config = BigQueryConfig.from_env()
    writer = BigQueryWriter(config)
    writer.create_tables_if_not_exist()
    return writer
