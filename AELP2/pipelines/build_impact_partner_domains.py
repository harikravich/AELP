#!/usr/bin/env python3
"""
Build partner→domain mapping from Impact MediaPartners table.

Reads:
  <project>.<dataset>.impact_media_partners

Writes:
  <project>.<dataset>.impact_partner_domains (partner_id STRING, partner STRING, domain STRING, root STRING)

Domain extraction:
  - Use top-level field Website
  - Use Properties[].Url for WEBSITE types
  - Normalize to hostname, drop scheme/path, downcase
  - Root = first label (leftmost token) stripped of common prefixes (www)
"""
from __future__ import annotations
import os, re, urllib.parse
from typing import Dict, Any, List
from google.cloud import bigquery


def to_host(url: str) -> str | None:
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    if '://' not in u:
        u = 'https://' + u
    try:
        host = urllib.parse.urlparse(u).hostname
        if host:
            return host.lower()
    except Exception:
        return None
    return None


def root_of(host: str) -> str:
    h = host.lower()
    if h.startswith('www.'):
        h = h[4:]
    # take left-most label as root (e.g., rakuten.com -> rakuten)
    return h.split('.')[0]


def main():
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    c = bigquery.Client(project=project)
    rows = list(c.query(f"SELECT * FROM `{project}.{dataset}.impact_media_partners`", location='us-central1').result())
    out: List[Dict[str, Any]] = []
    for r in rows:
        js = dict(r)
        pid = str(js.get('Id') or js.get('id') or '')
        name = js.get('Name') or js.get('name') or ''
        hosts = set()
        host = to_host(js.get('Website') or '')
        if host:
            hosts.add(host)
        props = js.get('Properties') or []
        if isinstance(props, list):
            for p in props:
                if isinstance(p, dict) and (p.get('Type') == 'WEBSITE'):
                    h = to_host(p.get('Url') or '')
                    if h:
                        hosts.add(h)
        for h in hosts:
            out.append({'partner_id': pid, 'partner': name, 'domain': h, 'root': root_of(h)})
    # Load to BQ
    table_id = f"{project}.{dataset}.impact_partner_domains"
    try:
        c.delete_table(table_id, not_found_ok=True)
    except Exception:
        pass
    schema = [
        bigquery.SchemaField('partner_id','STRING','REQUIRED'),
        bigquery.SchemaField('partner','STRING','NULLABLE'),
        bigquery.SchemaField('domain','STRING','REQUIRED'),
        bigquery.SchemaField('root','STRING','REQUIRED'),
    ]
    t = bigquery.Table(table_id, schema=schema)
    c.create_table(t)
    c.load_table_from_json(out, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')).result()
    print(f"Loaded {len(out)} rows into {table_id}")


if __name__ == '__main__':
    main()

