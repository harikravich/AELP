# Security Hardening Checklist (Phase 1)

IAM (BigQuery)
- Datasets: `${BIGQUERY_TRAINING_DATASET}`, `gaelp_users`
  - Grant: `roles/bigquery.dataViewer`, `roles/bigquery.jobUser` to runtime SA
  - Grant write only where needed (tables created by flows): `roles/bigquery.dataEditor` (narrow to target tables)
- Principle of least privilege: no `roles/editor` on project

Secrets
- Store Ads OAuth (client/secret/refresh) outside repo (Secret Manager)
- Rotate any committed secrets immediately; update service config
 - Repository remediation completed: sanitized `google_ads_oauth.json` and removed hardcoded GA4 client secret from code; OAuth now reads from env.

Network
- (Optional) VPC Service Controls perimeter for BigQuery; add project to perimeter

Audit
- Enable Admin Activity + Data Access logs for BigQuery; review weekly
- Run `python3 -m AELP2.pipelines.security_audit` (writes to `iam_audit`)

Flags & HITL
- Keep `GATES_ENABLED=1`, `AELP2_ALLOW_*` default `0`
- Require HITL API approval for any live mutations
