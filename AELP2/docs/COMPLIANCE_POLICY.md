# Compliance & Safety Policy (Marketing Systems)

Scope: creatives, landing pages, data collection, minors, platform policies.

Principles
- Consent: collect only with explicit user/parental consent; disclose purpose, storage, and usage.
- Minimization: store only what’s needed; prefer aggregates and pseudonymization.
- Transparency: plain-language notices on LPs; link to privacy policy; easy opt-out.
- Safety gates: all live mutations HITL; logs/audits for every change; rollback available.

Minors & Sensitive Signals
- Any teen/child-related features (e.g., Balance “insight preview”) must be consent-gated; no storage of non‑public data; default shadow mode.
- Public handles: hash client-side before transit; fetch only public/allowed signals; avoid scraping policies violations.

Platform Policies (Google/YouTube/Meta/TikTok)
- Pre‑lint all generated copy via moderation + Ads policy checks; avoid sensitive medical/mental health claims.
- Avoid implying diagnosis or guaranteed outcomes; avoid targeting minors.

Data Handling
- Ads/GA4 joins limited to KPI metrics; avoid PII in BQ tables; redact names; hashed identifiers.
- Retention: set retention policies on raw logs; archive after business purpose expires.

Approvals
- Legal review required for dynamic insight modules; security review for audience exports; brand/policy review for creative templates.

