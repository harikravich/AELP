#!/usr/bin/env python3
"""
Quick sanity tests for bandit_service.
Run as: python3 -m AELP2.tests.test_bandit_service
Exits 0 on success, 1 on failure.
"""
import sys
from AELP2.core.optimization.bandit_service import thompson_select, mabwiser_select, HAVE_MABWISER


def run():
    arms = [
        {'ad_id': 'a', 'impressions': 5000, 'clicks': 200},
        {'ad_id': 'b', 'impressions': 6000, 'clicks': 180},
        {'ad_id': 'c', 'impressions': 4000, 'clicks': 140},
    ]
    s1, ann1 = thompson_select(arms)
    assert s1.get('ad_id') in {'a','b','c'} and len(ann1) == 3
    if HAVE_MABWISER:
        s2, ann2 = mabwiser_select(arms)
        assert s2.get('ad_id') in {'a','b','c'} and len(ann2) == 3
    print('bandit_service sanity: OK')


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print('bandit_service sanity FAILED:', e)
        sys.exit(1)
