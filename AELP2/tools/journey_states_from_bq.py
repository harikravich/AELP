#!/usr/bin/env python3
"""
Optional: Pull journey states from BigQuery if available.
Reads env: GOOGLE_CLOUD_PROJECT, BIGQUERY_USERS_DATASET
Looks for tables: user_journeys, journey_touchpoints, conversion_events
Writes a tiny sample JSON with row counts to AELP2/reports/journey_states_probe.json
"""
from __future__ import annotations
import os, json
from google.cloud import bigquery


def main():
    project=os.getenv('GOOGLE_CLOUD_PROJECT'); dataset=os.getenv('BIGQUERY_USERS_DATASET')
    out={'project':project,'dataset':dataset,'tables':{},'note':'present=counts; missing=0'}
    if not project or not dataset:
        out['error']='Missing env GOOGLE_CLOUD_PROJECT or BIGQUERY_USERS_DATASET'
    else:
        client=bigquery.Client(project=project)
        for t in ['user_journeys','journey_touchpoints','conversion_events']:
            table=f"{project}.{dataset}.{t}"
            try:
                q=f"SELECT COUNT(1) c FROM `{table}`"
                rows=list(client.query(q).result())
                out['tables'][t]=int(rows[0].c)
            except Exception as e:
                out['tables'][t]=0
                out['tables'][t+'_error']=str(e)
    os.makedirs('AELP2/reports', exist_ok=True)
    path='AELP2/reports/journey_states_probe.json'
    open(path,'w').write(json.dumps(out,indent=2))
    print(path)

if __name__=='__main__':
    main()

