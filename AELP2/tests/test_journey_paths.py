#!/usr/bin/env python3
from AELP2.pipelines.journey_path_summary import compute_paths
from datetime import datetime, timedelta


def test_compute_paths_basic():
    now = datetime.utcnow()
    rows = [
        {'user_id': 'u1', 'session_start': now - timedelta(days=2), 'ch': 'search'},
        {'user_id': 'u1', 'session_start': now - timedelta(days=1), 'ch': 'display'},
        {'user_id': 'u1', 'session_start': now, 'ch': 'direct'},
        {'user_id': 'u2', 'session_start': now - timedelta(days=1), 'ch': 'search'},
        {'user_id': 'u2', 'session_start': now, 'ch': 'direct'},
    ]
    out = compute_paths(rows)
    # Expect a path for u1 and u2 and some transition probabilities
    assert any(o['path'] == 'search>display>direct' for o in out)
    assert any(o['path'] == 'search>direct' for o in out)
    assert any(o['transition_prob'] is not None for o in out)


if __name__ == '__main__':
    test_compute_paths_basic(); print('journey_paths test OK')

