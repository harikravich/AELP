# Security & Threat Model

Assets
- API keys (Ads/YouTube/GA4), SA credentials, BQ datasets, creative assets.

Threats
- Key leakage; unauthorized mutations; PII exposure; policy strikes; data poisoning.

Mitigations
- No client-side secrets; Cloud Run SA with least privilege; flags + HITL; publish paused; audit logs; GE DQ gates.
- Redaction of names; hashed IDs; consent gating; retention policies.

Monitoring
- Ops alerts on DQ failures, policy strikes, CAC breaches, API errors.

