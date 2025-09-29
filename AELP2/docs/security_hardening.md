Security Hardening (GCP + BigQuery)

Principles
- No credentials in repo (rotate any committed secrets immediately).
- Least privilege IAM: narrow dataset access to specific service accounts.
- Network controls: use VPC Service Controls (VPC‑SC) for BigQuery where feasible.
- Auditability: enable Admin Activity + Data Access logs.
- Data minimization: no PII; redact human‑readable names when not required.

Actions
1) Rotate any committed secrets
- Developer token, client secrets, refresh tokens in files should be considered compromised; rotate in Google Ads and GCP.
- Move secrets to environment or Secret Manager; never commit to git.

2) Create a service account for AELP2
- gcloud iam service-accounts create aelp2-train --display-name="AELP2 Training"
- Grant only dataset‑level BigQuery roles:
  - bq update --set_iam_policy to add roles/bigquery.dataEditor on `${BIGQUERY_TRAINING_DATASET}` for the SA
  - roles/bigquery.dataViewer for read‑only users

3) Dataset access policy
- Remove broad principals from `${BIGQUERY_TRAINING_DATASET}`; add only SA and specific users.
- Consider CMEK for BigQuery if required.

4) VPC Service Controls (optional, recommended)
- Create a VPC‑SC perimeter for BigQuery with your project; add egress rules only for necessary services.
- This reduces data exfiltration risk even if credentials leak.

5) Logging and monitoring
- Enable Admin Activity and Data Access logs for BigQuery and Google Ads usage.
- Set alerts on unusual query volumes or IAM changes.

6) Redaction and PII
- Ads loader redacts campaign_name by default (stores SHA256 in campaign_name_hash).
- GA4 loader imports only aggregates; for raw events via GA4 export, avoid ingesting user_pseudo_id or sensitive fields outside secure datasets.

7) Server hygiene (GCE)
- Use OS Login or IAP; disable password SSH; use SSH keys with short TTL.
- Restrict firewall to required egress only; no 0.0.0.0/0 ingress.
- Auto‑updates and unattended upgrades; lock down sudoers to specific admins.
- Keep ADC tokens restricted to least privilege on this host.

