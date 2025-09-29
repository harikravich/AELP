GA Data Atlas (Initial)

This Atlas summarizes high-signal GA4 export fields we use to derive KPIs and cohorts.

- Property export: `ga360-bigquery-datashare.analytics_308028264` (US)
- Window profiled: last ~180 days

## Core Events
- purchase (enrollments)
- form_submit (trial/flow signal; page_location used)
- begin_checkout / Checkout (checkout starts)
- page_view, window_loaded[_seven_seconds] (flow timing)

## High-signal Event Params on purchase
- `plan_code` (e.g., auraind, auracouple, auranoncreditpc, …)
- `cc` (campaign/offer code; e.g., a_22_50, extrasave3, a_24_social, …)
- `initial_payment_revenue` (numeric)
- `trial_length`, `subscription_length_days`, `term_length`
- `source_cookie`, `medium_cookie`, `campaign_cookie`, `landing_page_cookie`
- `impact_click_id[_cookie]`, `impact_ad_id[_cookie]`
- `page_location`, `page_referrer`

## Items on purchase
- `items.item_id` (offer_code), `items.item_name`, `items.item_variant`

## Cohort Logic (implemented)
- Trial signal: `form_submit` where `page_location` matches:
  `/enrollment/|/onboarding-path/|free[-_]?trial|start[-_]?trial|freetools|(^|[=_-])ft([0-9]|$)|ftupsell|/trial`
- D2P starts: purchase with no trial signal in the prior 14 days.
- Post‑trial subs: first purchase within 7 days after trial signal.
- Mobile subs (v1 heuristic): purchase where `cc` matches `(?i)mob|mobile|mobcart|app|ios|android` or `plan_code` matches similar tokens; `device.category` if present on purchase.

## BigQuery Artifacts (gaelp_training)
- `ga4_derived_daily` (date, enrollments, d2p_starts, post_trial_subs, mobile_subs)
- `ga4_derived_monthly` (month, …)
- `ga4_offer_code_daily` (date, plan_code, cc, source, medium, host, purchases, initial_payment_revenue)
- `cross_platform_kpi_daily` (view): Ads cost + GA conversions

## Joins & Cross-Platform Mapping
- GA↔Impact: `impact_click_id[_cookie]`
- GA source/medium normalized to canonical channels in the view layer
- Offer mapping via `plan_code`, `cc`, and `items.item_id`

## QA Pointers
- “Daily Enrollments” (Explore) ≈ `COUNTIF(event_name='purchase')` per day
- For 2025‑09‑15: GA purchases = 1,097 vs Pacer D2C = 1,186 (~92.5%)
- Tie‑outs CSVs in `AELP2/exports/`

> This Atlas will evolve as we add Meta/Impact and (optionally) a GA app export. Update `AELP2/config/ga4_mapping.yaml` to change cohort windows and rules.
