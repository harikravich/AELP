# Bandits Integration (MABWiser)

Policy
- Thompson Sampling by cell (Angle×Audience×Channel×LP×Offer) with Beta/Bernoulli for conversion‑rate, Gaussian for value if available.

Data
- Features: angle, audience, channel, lp, offer; metrics from `explore_cells` and Ads/GA4 joins.
- Persist: `bandit_posteriors` (mean, ci_low, ci_high, samples); `rl_policy_snapshots` (config + priors).

Loop
- Daily job reads last N days, updates posteriors, writes snapshots; selects arms for exploration share (10–15%).
- Writes proposals to `bandit_change_proposals` (shadow) for review; dashboard shows Promote/Kill calls.

Safety
- Respect per-change and daily caps; HITL approvals required for budget changes and publishes.

