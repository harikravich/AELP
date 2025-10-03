#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
source .env >/dev/null 2>&1 || true

# Phase 2 baseline
python3 AELP2/tools/sim_fidelity_campaigns_temporal.py || true

# Phase 2 v2 (decay + mixture + creative-age)
AELP2_DECAY_HL=${AELP2_DECAY_HL:-7} AELP2_PI_SHRINK=${AELP2_PI_SHRINK:-0.8} \
python3 AELP2/tools/sim_fidelity_campaigns_temporal_v2.py || true

# Rolling + forecast
python3 AELP2/tools/fidelity_eval_roll.py || true
python3 AELP2/tools/forward_forecast.py || true

# Hourly multipliers and RL shadow score
python3 AELP2/tools/hourly_multipliers.py || true
python3 AELP2/tools/rl_shadow_score.py || true

# One-pager
python3 AELP2/tools/render_sim_onepager.py || true

echo "Done. Reports in AELP2/reports" >&2

