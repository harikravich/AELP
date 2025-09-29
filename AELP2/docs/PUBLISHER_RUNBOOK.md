# Creative Publisher Runbook

Enqueue
- POST `/api/control/creative/enqueue` with payload (RSA/PMax/Video);
- Inspect row in `creative_publish_queue` (status=queued).

Publish (DRY + paused)
- POST `/api/control/creative/publish` (or schedule Cloud Run job)
- Inspect `creative_publish_log` for platform IDs; all assets/ads should be PAUSED.

Rollback
- POST `/api/control/creative/rollback` with platform IDs to pause/remove.

Troubleshooting
- Missing env: GOOGLE_ADS_* and MCC login
- Policy disapproval: read topics; adjust copy/assets; re‑enqueue

