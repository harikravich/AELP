GA4 Auth Setup

Status: Ready (requires external grant)

Two supported auth paths:
- Service Account (recommended): grant the Compute VM SA `analytics.readonly` to the GA4 property and link it in GA admin. Then set `GOOGLE_APPLICATION_CREDENTIALS` or run on GCE/Cloud Run with the SA.
- OAuth Refresh Token (fallback): provide `GA4_OAUTH_REFRESH_TOKEN` and `GA4_OAUTH_CLIENT_ID/SECRET` as env. The loaders will use user OAuth for the GA4 Data API.

Verification:
`python3 AELP2/pipelines/ga4_permissions_check.py --dry_run`

Guardrails:
- No writes to GA; only read scopes.
- If neither path is available, loaders run in dry‑run and log guidance.

