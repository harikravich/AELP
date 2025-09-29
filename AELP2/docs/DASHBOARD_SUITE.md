Dashboard Suite Plan (Creative Center, Training Center, Exec Dashboard)

Goals
- Provide a sleek, usable interface for Aura’s marketing team to operate creatives, monitor training, and review top‑level KPIs.
- Unify approvals/HITL, AB testing, and platform data (Ads/GA4) with production safety and redaction.

Sections
- Creative Center
  - Creative Library: import existing Google Ads creatives (ads_ad_performance) with hashes/redaction;
    searchable by campaign/ad_group/keyword/device.
  - Approvals & HITL: view/approve queued actions; policy issues surfaced inline; audit trail from safety_events.
  - AB Hub: configure/monitor creative experiments; write to ab_experiments; lift metrics (CTR/CVR/ROAS, CAC).
  - Generation (shadow first): AI‑assisted variants inspired by winning creatives (flags: ALLOW_REAL_CREATIVE, AELP2_SUBAGENTS_SHADOW).
  - Assets: handle images/video safely (hashed refs; no PII) and map to platform creatives.

- Training Center
  - Episodes & Metrics: win_rate/CAC/ROAS/spend/epsilon over time; calibration status (scale/offset, KS/MSE);
    safety events timeline; fidelity evaluation results.
  - Subagents View: proposals per subagent (drift/budget/creative/targeting); approvals; outcomes; quotas.
  - Live Signals: GA4 aggregates and Ads aggregates for the same window; sim vs IRL overlays.
  - Controls: flag toggles (shadow mode, HITL non-blocking, cadence), guarded by RBAC.

  - Auctions Monitor (new):
    - Live win‑rate gauge vs target band; bid distribution histogram; price‑paid (2nd price) trend.
    - Floor ratio over time; auto‑tuner adjustments; nowin‑guard activations; epsilon trend.
    - Channel/device heatmap for win‑rate and price‑paid; top contexts by volume.
    - Bid Replay inspector: sample recent decisions with features (context hash), our_bid, floor, competitor_top_bid, price_paid, safety flags — “why this bid?”.
    - Calibration panel: signature detection, KS/MSE, current floor ratio, target band, last recalibration timestamps.

  - Learning Avatar (optional, toggle):
    - Purpose: engaging, intuitive depiction of agent progress without replacing real KPIs.
    - Mapping (example):
      - Win-rate → stride length/speed; CAC → avatar weight/drag; ROAS → trail color (red→green);
      - Epsilon → jitter/wobble; Safety events → obstacles/stoplights; Calibration score → terrain smoothness;
      - Fidelity error (MAPE/KS) → fog opacity; Subagent activity → helper icons orbiting avatar.
    - Implementation: Start simple 2D animation (Plotly frames/Canvas) in Streamlit, toggle on/off.
      Graduate to 3D (Three.js) if useful and performant.

- Exec Dashboard
  - KPIs: CAC, ROAS, spend, conversions, win_rate, impression share; trends and error bands; safety incident counts.
  - Attribution Summary: AOV vs LTV basis; lag windows; touchpoint/conversion counts.
  - Budget Split: by channel/segment/device; rebalancing proposals.
  - Auction Insights (new): competitor overlap/position metrics (from Ads Auction Insights) with trends.

Data Sources & Views (BigQuery)
- training_episodes (AELP2 telemetry)
- safety_events (HITL/policy/gates)
- ab_experiments (creative experiments)
- fidelity_evaluations (sim vs IRL metrics)
- ads_campaign_performance, ads_keyword_performance, ads_search_terms, ads_ad_performance, ads_conversion_actions
- ga4_aggregates and GA4 native export (recommended)
- Views available:
  - training_episodes_daily: by date, with CAC/ROAS and win_rate
  - ads_campaign_daily: by date (+cac, roas, impression_share, ctr/cvr)
  - subagents_daily: proposals by subagent and event_type per day
  - ga4_daily (if GA4 aggregates present) and ga4_export_daily (if native export dataset configured)
  - segment/device slices (if available in telemetry)
  - bidding_events (raw), bidding_events_minutely (agg), bidding_events_by_channel_device (agg)
  - ads_auction_insights (if available)

Tech & Rollout (Agreed FE Stack)
- Frontend: React + Next.js (App Router) + TypeScript
- UI: shadcn/ui + Tailwind CSS
- State/data: TanStack Query, Zustand (UI state)
- Charts: Recharts or Visx; Mapbox for geo if needed
- Auth: NextAuth with Google Workspace SSO; RBAC via roles (viewer/approver/editor/admin)
- API: Next.js route handlers → BigQuery (read), secure feature flags, HITL APIs
- Live: SSE/WebSocket for near‑real‑time episodes/safety updates (optional)
  - Auctions Monitor: SSE (or short‑poll) for recent bidding_events; throttle updates; fall back to per‑minute aggregates.
- Deployment: Vercel/Cloud Run; env via secrets; CI with checks

App Structure (Next.js)
- /creative-center: library, approvals, AB hub, (shadow) AI variants
- /training-center: episodes/metrics, safety timeline, calibration/fidelity, subagents
- /exec: KPI trends, attribution summary, budget splits
- /auth, /admin: RBAC, feature flags, platform health

Notes
- Reuse BQ views (`training_episodes_daily`, `ads_campaign_daily`, `subagents_daily`, GA4 views)
- Keep PII out: redacted hashes for free‑text fields

Security & RBAC
- Roles: viewer (read), approver (HITL), editor (configure flags/AB), admin (platform auth/IAM).
- Enforce redaction (no PII, hash fields in BQ); audit logs from safety_events.

Approvals & Safety Integration
- Use safety.hitl queue and BigQuery events; ensure non‑blocking flows for training; hard blocks for live creative/targeting changes.

Open Tasks
- Build/import ads_ad_performance loader (creatives) if not present; add BQ views.
- Implement Streamlit prototypes with RBAC/auth patterns.
- Wire AB and approvals into Creative Center; tie to adapters in shadow.
