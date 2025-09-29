# Use Case: Aura Balance (Parental Controls & Digital Wellness)

Objective
- Scale to $100k+/day at CAC ≤ $80 (channel‑specific targets below) by exploring multiple angles across audiences/channels/LPs/offers, learning fast, and ramping with guardrails.

Personas & JTBD
- Parents 25–54 (iOS priority), concerned about online balance, safe gaming, social interactions, scams/fraud.
- Jobs: understand child's online day/night balance, detect risky behaviors, set healthy guardrails without snooping, get actionable alerts.

Primary KPIs
- New customers (trial start, purchase); CAC; ROAS; LTV(90) proxy.
- Secondary: CTR, CVR, LP funnel completion, brand search lift, audience growth.

Conversion Actions & Values (Google Ads)
- Trial start (primary), Purchase, Subscription; include_in_conversions=true for KPI actions; map values for ROAS.

Channels & CAC Targets (initial)
- Search Brand ≤ $70; Search Non‑Brand ≤ $80; YouTube Shorts/Instream ≤ $95; Discovery ≤ $90; PMax ≤ $85.

Angles & Hooks (examples)
- Balance‑Safety: “See the whole picture, not just screen time.”
- Safe Gaming: “Know when gaming crosses into harm—without snooping.”
- Digital Wellness: “Healthy online day & night balance for your family.”
- Scam/Fraud: “Protect your kids from scams, fake friends, and fraud.”

Creative Requirements (Google Ads)
- RSA: 8–15 headlines (≤30 chars), 4 descriptions (≤90), 1–2 paths, final URL; policy‑safe copy.
- PMax Asset Group: text assets (H1 ≤30, long headline ≤90, desc ≤90), 1:1, 1.91:1 images; optional YouTube video asset.
- YouTube: 6–15s Shorts & 15–30s in‑stream; 9:16, 1:1, 16:9 variants; strong hook in first 2s; on‑screen captions; end slate with CTA.

LP Strategy
- Variant A (Fast Path): Ad → Order flow; short hero; benefits; trust badges; price anchor; sticky CTA.
- Variant B (Insight Path): Ad → Balance LP with consent‑gated “Insight Preview” (public/safe signals only, shadow by default), then CTA.
- Blocks: hero, benefits, trust, FAQ, pricing anchor, sticky CTA, social proof; mobile‑first.

Compliance (Balance)
- No sensitive medical/diagnostic claims; avoid implying mental health diagnosis.
- Consent‑gate any dynamic insight; pseudonymize handles; shadow mode until Legal approves storage.

Exploration Cells (Balance‑focused; initial 12)
1) Balance‑Safety | Parents iOS | Search‑Brand | /balance‑v2 | trial
2) Balance‑Safety | Parents iOS | Search‑NB | /balance‑v2 | trial
3) Balance‑Safety | Parents Teens | YouTube Shorts | /balance‑v2 | trial
4) Safe Gaming | Parents iOS | YouTube Shorts | /balance‑v2 | trial
5) Digital Wellness | Parents Android | Search‑NB | /balance‑v2 | trial
6) Balance‑Safety | Parents iOS (CA) | Search‑NB | /balance‑v2 | trial
7) Balance‑Safety | Parents iOS | Discovery | /balance‑v2 | trial
8) Balance‑Safety | Parents iOS | PMax | /balance‑v2 | trial
9) Safe Gaming | Parents Teens | YouTube Instream | /balance‑v2 | trial
10) Digital Wellness | Parents 25–54 | YouTube Shorts | /balance‑v2 | trial
11) Balance‑Safety | Parents iOS | Search‑NB | /balance‑insight‑v1 | trial
12) Balance‑Safety | Parents iOS | PMax | /balance‑insight‑v1 | trial

Budget & Caps
- Exploration budget: 10–15% daily; $50–$150/day per cell; per‑change ≤5%; daily ≤10%.

Tests (A/B)
- Creative: winner vs 3 LLM‑guardrailed variants (headlines, CTAs) → CTR/CVR and CAC.
- LP: /balance‑v2 vs /balance‑insight‑v1 (mobile‑first changes: sticky CTA, short hero vs long) → CVR uplift.
- Offer framing: “14‑day trial” vs “Get started in 2 minutes” (same price/terms).

Measurement & Halo
- GA4 lag‑aware conversions; funnel events; SRM checks for A/B.
- Weekly GeoLift on YouTube Shorts bursts (selected geos) to measure brand‑search lift; adjust ramp ROI.

Data & Tables Used
- Ads: ads_campaign_performance, ads_ad_performance, conversion_actions/stats.
- GA4: ga4_daily, ga4_lagged_daily.
- Experiments: ab_assignments, ab_exposures, ab_metrics_daily.
- Explore/RL: explore_cells, bandit_posteriors, rl_policy_snapshots.
- LP: lp_tests, funnel_dropoffs.
- Halo: halo_experiments, halo_reads_daily.

Execution Plan (2 weeks)
Week 1
- Launch 12 exploration cells (above) at $50–$100/day; publish creatives paused → HITL unpause; log to queue/log.
- Ship LP v2 and insight v1 templates; wire A/B 50/50; GA4 funnels; set KPI IDs; run MMM headroom APIs.
- Add weekly GeoLift job config; pick test geos; schedule.
Week 2
- Evaluate posteriors and LP results (48–72h cadence); promote 2–3 winners under caps; request +$2–$5k/day on Search Brand/YouTube with halo check.
- Generate next 6 cells based on learnings; retire weak cells; iterate creatives.

Owner Map (initial)
- Creative: Growth Eng + Design; LPs: Growth Eng + Design; Data/Models: DS; Dashboard/API: FE/BE; Ops: SRE; Legal: consent/dynamic module.

