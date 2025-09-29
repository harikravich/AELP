# RL Selector APIs (Draft)

## POST /rl/v1/proposals
Submit selector RL proposals for a campaign day.

- Body schema: `AELP2/docs/schemas/rl_proposals.schema.json`
- Response: 200 OK, stored under `rl_proposals` table

## POST /rl/v1/actions/approve
Approve and apply a subset of proposals (Controlled Mode).

- Body fields: proposal_id, approved_actions[], approver, comment
- Response: applied action ids; errors if any

## GET /rl/v1/status?campaign_id=...&date=...
Returns last proposals, approvals, outcomes.

- Includes simulator predictions alongside actuals for transparency.

## Webhooks
- `rl/proposal_created`, `rl/actions_applied`, `rl/rollback_triggered`

---

# Reporting Exports
- Daily diff (CSV): proposals vs predicted vs actual → `AELP2/reports/rl_daily_diff_YYYYMMDD.csv`
- Creative leaderboard (CSV/JSON): purchases, CAC, age, freq, template_id
