#!/usr/bin/env python3
"""
Build multi-touch attribution from GA4 export:
  - Position-based 40/20/40 daily credits per channel: <project>.<dataset>.ga4_attribution_daily_pb
  - Markov removal effects (30d aggregate): <project>.<dataset>.ga4_markov_contrib_30d

Approach
  - Construct per-user paths of channels (source/medium) for each purchase, using a 30-day lookback.
  - Position-based weights: 0.4 to first, 0.4 to last, remaining 0.2 split equally among middle touches.
  - Markov: aggregate paths (Start -> ... -> Conversion) and compute removal effects in Python.

Env:
  - GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  - GA4_EXPORT_DATASET (e.g., ga360-bigquery-datashare.analytics_308028264)
"""
import os
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from google.cloud import bigquery


def build_paths_sql(export_ds: str, days: int = 60) -> str:
    """SQL to build per-purchase channel paths for the last N days.
    Channels derive from source_cookie/medium_cookie.
    """
    return f"""
    WITH ev AS (
      SELECT
        user_pseudo_id,
        TIMESTAMP_MICROS(event_timestamp) AS ts,
        PARSE_DATE('%Y%m%d', event_date) AS d,
        (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(event_params) ep WHERE ep.key='source_cookie') AS source,
        (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(event_params) ep WHERE ep.key='medium_cookie') AS medium,
        event_name,
        _TABLE_SUFFIX AS sfx
      FROM `{export_ds}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY))
                              AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
    ), intra AS (
      SELECT
        user_pseudo_id,
        TIMESTAMP_MICROS(event_timestamp) AS ts,
        DATE(TIMESTAMP_MICROS(event_timestamp)) AS d,
        (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(event_params) ep WHERE ep.key='source_cookie') AS source,
        (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(event_params) ep WHERE ep.key='medium_cookie') AS medium,
        event_name,
        _TABLE_SUFFIX AS sfx
      FROM `{export_ds}.events_intraday_*`
      WHERE _TABLE_SUFFIX = FORMAT_DATE('%Y%m%d', CURRENT_DATE())
    ), base AS (
      SELECT * FROM ev UNION ALL SELECT * FROM intra
    ), touches AS (
      SELECT user_pseudo_id, ts, d,
             CONCAT(IFNULL(source,'(none)'), '/', IFNULL(medium,'(none)')) AS channel,
             event_name
      FROM base
      WHERE (source IS NOT NULL OR medium IS NOT NULL)
    ), purchases AS (
      SELECT user_pseudo_id, ts AS purchase_ts, d AS purchase_date
      FROM base
      WHERE event_name='purchase'
    )
    SELECT p.user_pseudo_id, p.purchase_date,
           ARRAY_AGG(t.channel ORDER BY t.ts) AS path
    FROM purchases p
    JOIN touches t
      ON t.user_pseudo_id = p.user_pseudo_id
     AND t.ts BETWEEN TIMESTAMP_SUB(p.purchase_ts, INTERVAL 30 DAY) AND p.purchase_ts
    GROUP BY p.user_pseudo_id, p.purchase_date
    """


def compute_markov_contrib(paths: List[List[str]]) -> Dict[str, float]:
    """Compute Markov removal effects on aggregated paths.
    States: START -> channels... -> CONV. Removal effect = (baseline - removed)/baseline.
    Returns channel -> contribution (unnormalized); caller will normalize to shares.
    """
    # Build transition counts
    trans = Counter()
    conv_count = 0.0
    for p in paths:
        seq = ['START'] + (p or []) + ['CONV']
        for i in range(len(seq)-1):
            trans[(seq[i], seq[i+1])] += 1.0
        conv_count += 1.0
    # Compute transition probabilities
    from collections import defaultdict
    out = defaultdict(float)
    nxt = defaultdict(float)
    # adjacency
    adj = defaultdict(list)
    totals = defaultdict(float)
    for (a,b), c in trans.items():
        totals[a] += c
        adj[a].append((b, c))
    P = { (a,b): (c / totals[a]) for (a,b), c in trans.items() if totals[a] > 0 }

    def conv_prob(remove: str | None = None) -> float:
        # simple power-iteration on absorbing chain with removal (edges via 'remove' eliminated)
        states = set([a for a,_ in trans] + [b for _,b in trans])
        states.discard('CONV')
        states.add('START')
        # One-step probability of conversion from each state
        pi = {s: 0.0 for s in states}
        # iterate a few steps to approximate
        for _ in range(6):
            new = {s: 0.0 for s in states}
            for s in states:
                for b, _c in adj.get(s, []):
                    if remove and b == remove:
                        continue
                    pr = P.get((s,b), 0.0)
                    if b == 'CONV':
                        new[s] += pr
                    else:
                        new[s] += pr * pi.get(b, 0.0)
            pi = new
        return pi.get('START', 0.0)

    base = conv_prob()
    contrib = {}
    chans = sorted({b for (_,b) in trans if b not in ('CONV','START')})
    for c in chans:
        rem = conv_prob(remove=c)
        effect = max(base - rem, 0.0)
        contrib[c] = effect
    return contrib


def main():
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    export_ds = os.environ.get('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')
    days = int(os.getenv('GA4_ATTRIB_DAYS', '60'))
    bq = bigquery.Client(project=project)

    # 1) Build paths in BQ (temporary result)
    sql = build_paths_sql(export_ds, days=days)
    job = bq.query(sql, location='US')
    paths = list(job.result())

    # 2) Position-based 40/20/40 per day
    daily = defaultdict(lambda: defaultdict(float))  # date -> channel -> credits
    all_paths = []
    for r in paths:
        pd = r['purchase_date']
        raw = r['path'] or []
        arr = []
        for x in raw:
            if isinstance(x, dict) and 'channel' in x:
                arr.append(x['channel'])
            else:
                # array of strings case
                arr.append(str(x))
        arr = [c if c else '(none)/(none)' for c in arr]
        # de-dup consecutive duplicates
        dedup = []
        for c in arr:
            if not dedup or dedup[-1] != c:
                dedup.append(c)
        L = len(dedup)
        if L == 0:
            continue
        if L == 1:
            daily[pd][dedup[0]] += 1.0
        else:
            daily[pd][dedup[0]] += 0.4
            daily[pd][dedup[-1]] += 0.4
            if L > 2:
                mid = 0.2 / (L - 2)
                for c in dedup[1:-1]:
                    daily[pd][c] += mid
        all_paths.append(dedup)

    # Write ga4_attribution_daily_pb
    rows = []
    for d, m in daily.items():
        for ch, v in m.items():
            rows.append({'date': d.isoformat(), 'channel': ch, 'credits': float(v)})
    schema = [
        bigquery.SchemaField('date','DATE','REQUIRED'),
        bigquery.SchemaField('channel','STRING','REQUIRED'),
        bigquery.SchemaField('credits','FLOAT','REQUIRED'),
    ]
    tbl = f"{project}.{dataset}.ga4_attribution_daily_pb"
    try:
        bq.delete_table(tbl, not_found_ok=True)
    except Exception:
        pass
    t = bigquery.Table(tbl, schema=schema)
    t.time_partitioning = bigquery.TimePartitioning(field='date')
    bq.create_table(t)
    bq.load_table_from_json(rows, tbl, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')).result()

    # 3) Markov removal (30d aggregate)
    contrib = compute_markov_contrib(all_paths)
    total = sum(contrib.values()) or 1.0
    rows2 = [{'channel': k, 'contribution': float(v), 'share': float(v/total)} for k,v in contrib.items()]
    schema2 = [
        bigquery.SchemaField('channel','STRING','REQUIRED'),
        bigquery.SchemaField('contribution','FLOAT','REQUIRED'),
        bigquery.SchemaField('share','FLOAT','REQUIRED'),
    ]
    tbl2 = f"{project}.{dataset}.ga4_markov_contrib_30d"
    try:
        bq.delete_table(tbl2, not_found_ok=True)
    except Exception:
        pass
    t2 = bigquery.Table(tbl2, schema=schema2)
    bq.create_table(t2)
    bq.load_table_from_json(rows2, tbl2, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')).result()
    print(f"Wrote {len(rows)} PB daily rows and {len(rows2)} Markov contrib rows")


if __name__ == '__main__':
    main()
