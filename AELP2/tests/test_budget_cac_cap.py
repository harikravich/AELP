#!/usr/bin/env python3
import os
import sys
import types
from unittest import mock


def run_with_mocks():
    # Prepare env
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'proj'
    os.environ['BIGQUERY_TRAINING_DATASET'] = 'ds'
    os.environ['GOOGLE_ADS_LOGIN_CUSTOMER_ID'] = '1234567890'
    os.environ['GOOGLE_ADS_CUSTOMER_ID'] = '1234567890'
    os.environ['AELP2_CAC_CAP'] = '50.0'

    # Import module
    import importlib
    mod = importlib.import_module('AELP2.core.optimization.budget_orchestrator')

    # Patch BQ client and helpers
    class FakeBQ:
        def query(self, *a, **k):
            class _R:
                def result(self_inner):
                    class row:
                        bad = 0
                        s = 1000
                        age = 1
                        cac = 120.0
                    return [row()]
            return _R()

    with mock.patch('AELP2.core.optimization.budget_orchestrator.bigquery.Client', return_value=FakeBQ()):
        with mock.patch('AELP2.core.optimization.budget_orchestrator.get_latest_allocation', return_value={'proposed_daily_budget': 100.0, 'diagnostics': '{"uncertainty_pct":0.2}'}):
            with mock.patch('AELP2.core.optimization.budget_orchestrator.get_recent_spend', return_value=700.0):
                with mock.patch('AELP2.core.optimization.budget_orchestrator.get_top_campaign_ids', return_value=['C1']):
                    with mock.patch('AELP2.core.optimization.budget_orchestrator.estimate_campaign_cac', return_value=120.0):
                        captured = {}
                        def fake_run(cmd, check=False, env=None):
                            captured['env'] = dict(env or {})
                            class R: returncode = 0
                            return R()
                        with mock.patch('subprocess.run', side_effect=fake_run):
                            mod.main()
                        notes = captured.get('env', {}).get('AELP2_PROPOSAL_NOTES', '{}')
                        assert 'cap_reason' in notes and 'cac_estimates' in notes


def test_budget_cap_notes():
    run_with_mocks()


if __name__ == '__main__':
    test_budget_cap_notes(); print('budget_cac_cap test OK')

