# A/B Assignment Service Spec

Goal
- Assign units (ga4_client_id/user_id/cookie) to variants deterministically; log assignment and exposures; support SRM checks.

API
- `POST /api/ab/assign { experiment, unit_id, unit_type, context? }` → `{ variant, assigned }`

Method
- Namespace per experiment; hash(unit_id + salt) → [0,1), map to variant bands (default 50/50).
- Sticky: return same variant for same unit_id.
- Log assignment to `ab_assignments` (timestamp, unit_type, context).
- Exposures logged via `/api/ab/exposure` on first view/click.

SRM
- Nightly compute expected vs observed splits per experiment; red badge if p<0.01.

Security
- Server‑only; no secrets in client; throttle abuse.

