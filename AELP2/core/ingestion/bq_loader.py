"""
BigQuery Loader utilities for AELP2.

Strict behavior:
- No hardcoded project/dataset; read from env.
- Fail-fast with clear errors; no dummy data.
"""

import os
import logging
import time
from typing import List, Dict, Any

try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound, Forbidden
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    BIGQUERY_AVAILABLE = True
except ImportError as e:
    bigquery = None
    NotFound = None
    Forbidden = None
    ServiceAccountCredentials = None
    BIGQUERY_AVAILABLE = False
    IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


class BQIngestionError(Exception):
    pass


def get_bq_client() -> "bigquery.Client":
    """Create a BigQuery client.

    Honors optional env var `AELP2_BQ_CREDENTIALS` to use a dedicated
    service account JSON for BigQuery, separate from GOOGLE_APPLICATION_CREDENTIALS.
    """
    if not BIGQUERY_AVAILABLE:
        raise BQIngestionError(
            f"google-cloud-bigquery not installed: {IMPORT_ERROR}. Install with pip."
        )
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise BQIngestionError("GOOGLE_CLOUD_PROJECT env var is required")

    # Prefer explicit BigQuery SA key if provided
    creds_path = os.getenv("AELP2_BQ_CREDENTIALS")
    if creds_path:
        creds_path = os.path.expanduser(creds_path)
        if not ServiceAccountCredentials:
            raise BQIngestionError("google.oauth2.service_account not available to load AELP2_BQ_CREDENTIALS")
        if not os.path.isfile(creds_path):
            raise BQIngestionError(f"AELP2_BQ_CREDENTIALS path not found: {creds_path}")
        creds = ServiceAccountCredentials.from_service_account_file(creds_path)
        logger.info("BigQuery auth: using service account from AELP2_BQ_CREDENTIALS")
        return bigquery.Client(project=project, credentials=creds)

    # Use GCE metadata service account explicitly (ignores GOOGLE_APPLICATION_CREDENTIALS)
    use_gce = os.getenv("AELP2_BQ_USE_GCE", "").lower() in ("1", "true", "yes")
    if use_gce:
        old = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
        try:
            client = bigquery.Client(project=project)
            logger.info("BigQuery auth: using GCE metadata service account (AELP2_BQ_USE_GCE=1)")
            return client
        finally:
            if old is not None:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old

    # Default to ADC / GOOGLE_APPLICATION_CREDENTIALS
    return bigquery.Client(project=project)


def ensure_dataset(client: "bigquery.Client", dataset_id: str) -> None:
    try:
        client.get_dataset(dataset_id)
    except NotFound:
        ds = bigquery.Dataset(dataset_id)
        loc = os.getenv("BIGQUERY_DATASET_LOCATION")
        if loc:
            ds.location = loc
        client.create_dataset(ds)
        logger.info(f"Created dataset: {dataset_id}")
    except Exception as e:
        msg = str(e)
        if "Access Denied" in msg or "Permission" in msg or (Forbidden and isinstance(e, Forbidden)):
            raise BQIngestionError(
                "403 on BigQuery dataset access. Grant your credentials BigQuery access or set AELP2_BQ_CREDENTIALS. "
                "Recommended: grant roles/bigquery.user and roles/bigquery.dataEditor (or bigquery.admin) on the project, "
                "or add dataset-level BigQuery Data Owner to the service account."
            ) from e
        raise


def ensure_table(client: "bigquery.Client", table_id: str, schema: List["bigquery.SchemaField"], partition_field: str = None) -> None:
    try:
        client.get_table(table_id)
    except NotFound:
        table = bigquery.Table(table_id, schema=schema)
        if partition_field:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY, field=partition_field
            )
        client.create_table(table)
        logger.info(f"Created table: {table_id}")


def insert_rows(client: "bigquery.Client", table_id: str, rows: List[Dict[str, Any]]) -> None:
    """Insert rows with chunking and retries.

    Controlled by env:
      - AELP2_BQ_INSERT_BATCH_SIZE (default: 500 rows)
      - AELP2_BQ_INSERT_RETRIES (default: 3)
      - AELP2_BQ_INSERT_RETRY_BASE_MS (default: 500)
    """
    if not rows:
        return

    try:
        batch_size = int(os.getenv("AELP2_BQ_INSERT_BATCH_SIZE", "500"))
    except Exception:
        batch_size = 500
    try:
        max_retries = int(os.getenv("AELP2_BQ_INSERT_RETRIES", "3"))
    except Exception:
        max_retries = 3
    try:
        base_ms = int(os.getenv("AELP2_BQ_INSERT_RETRY_BASE_MS", "500"))
    except Exception:
        base_ms = 500

    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        attempt = 0
        while True:
            try:
                errors = client.insert_rows_json(table_id, chunk)
                if errors:
                    raise BQIngestionError(f"BQ insert errors: {errors}")
                break
            except Exception as e:
                if attempt >= max_retries:
                    raise BQIngestionError(
                        f"Failed to insert chunk {i//batch_size+1} ({len(chunk)} rows) into {table_id} after {max_retries} retries: {e}"
                    )
                backoff = (base_ms / 1000.0) * (2 ** attempt)
                logger.warning(
                    f"Insert failed for {table_id} (chunk {i//batch_size+1}, size={len(chunk)}). Retry {attempt+1}/{max_retries} in {backoff:.2f}s: {e}"
                )
                time.sleep(backoff)
                attempt += 1
