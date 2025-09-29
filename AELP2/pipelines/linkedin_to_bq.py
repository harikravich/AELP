#!/usr/bin/env python3
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

def ensure_table(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.linkedin_ad_performance"
    try:
        bq.get_table(table_id)
        return
    except NotFound:
        pass
    schema = [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("adgroup_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ad_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("impressions", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("clicks", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("cost", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("conversions", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("revenue", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ctr", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("cvr", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("avg_cpc", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("name_hash", "STRING", mode="NULLABLE"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="date")
    bq.create_table(table)
    print(f"Created {table_id}")

def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    ensure_table(bq, project, dataset)
    print('LinkedIn schema ensured.')

if __name__ == '__main__':
    main()

