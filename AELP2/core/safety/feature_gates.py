#!/usr/bin/env python3
"""
Feature gates / HITL flags wrapper.

Reads environment variables and optional local JSON to decide if an action is allowed.
This is a minimal GrowthBook-like gate without network calls.

Env vars:
- GATES_ENABLED=1 to enable gate checks (default: 1)
- GATES_FLAGS_JSON=/path/to/flags.json (optional; {"flag_name": true/false})
- AELP2_ALLOW_GOOGLE_MUTATIONS=0|1 (budget changes)
- AELP2_ALLOW_BANDIT_MUTATIONS=0|1 (creative changes)

Actions (string keys):
- apply_google_budget
- apply_creative_change
"""
import json
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_flags() -> dict:
    path = os.getenv('GATES_FLAGS_JSON')
    if path and os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def is_action_allowed(action: str) -> bool:
    if os.getenv('GATES_ENABLED', '1') != '1':
        return True
    flags = _load_flags()
    # Env-level overrides for core actions
    if action == 'apply_google_budget':
        if os.getenv('AELP2_ALLOW_GOOGLE_MUTATIONS', '0') != '1':
            return False
    if action == 'apply_creative_change':
        if os.getenv('AELP2_ALLOW_BANDIT_MUTATIONS', '0') != '1':
            return False
    # Optional flag file
    if action in flags:
        return bool(flags[action])
    return True


def gate_reason(action: str) -> str:
    if action == 'apply_google_budget' and os.getenv('AELP2_ALLOW_GOOGLE_MUTATIONS', '0') != '1':
        return 'AELP2_ALLOW_GOOGLE_MUTATIONS=0'
    if action == 'apply_creative_change' and os.getenv('AELP2_ALLOW_BANDIT_MUTATIONS', '0') != '1':
        return 'AELP2_ALLOW_BANDIT_MUTATIONS=0'
    return 'flag_denied_or_unknown'

