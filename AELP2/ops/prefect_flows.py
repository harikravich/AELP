#!/usr/bin/env python3
"""
Prefect flows scaffolding for MMM, Bandit, Uplift, and Opportunity Scanner.

Runs existing modules as tasks. If Prefect is not installed, prints instructions.
"""
import os
import subprocess
import sys
import json

try:
    from prefect import flow, task
    HAVE_PREFECT = True
except Exception:
    HAVE_PREFECT = False


def _run(cmd):
    print(f"[flow] running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


if HAVE_PREFECT:

    @task(retries=2, retry_delay_seconds=60)
    def mmm_v1():
        return _run([sys.executable, '-m', 'AELP2.pipelines.mmm_service'])

    @task(retries=2, retry_delay_seconds=60)
    def mmm_lightweightmmm():
        return _run([sys.executable, '-m', 'AELP2.pipelines.mmm_lightweightmmm'])

    @task(retries=2, retry_delay_seconds=60)
    def robyn_validator():
        return _run([sys.executable, '-m', 'AELP2.pipelines.robyn_validator', '--dry_run'])

    @task(retries=2, retry_delay_seconds=60)
    def channel_attribution_weekly():
        return _run([sys.executable, '-m', 'AELP2.pipelines.channel_attribution_r', '--dry_run'])

    @task(retries=2, retry_delay_seconds=60)
    def journeys_populate():
        return _run([sys.executable, '-m', 'AELP2.pipelines.journeys_populate'])

    @task(retries=2, retry_delay_seconds=60)
    def uplift_eval():
        return _run([sys.executable, '-m', 'AELP2.pipelines.uplift_eval'])

    @task(retries=2, retry_delay_seconds=60)
    def propensity_scores():
        return _run([sys.executable, '-m', 'AELP2.pipelines.propensity_uplift'])

    @task(retries=2, retry_delay_seconds=60)
    def segments_to_audiences():
        return _run([sys.executable, '-m', 'AELP2.pipelines.segments_to_audiences'])

    @task(retries=2, retry_delay_seconds=60)
    def training_posthoc():
        return _run([sys.executable, '-m', 'AELP2.pipelines.training_posthoc_reconciliation'])

    @task(retries=2, retry_delay_seconds=60)
    def youtube_reach_planner():
        return _run([sys.executable, '-m', 'AELP2.pipelines.youtube_reach_planner'])

    @task(retries=2, retry_delay_seconds=60)
    def kpi_consistency_check():
        return _run([sys.executable, '-m', 'AELP2.pipelines.kpi_consistency_check'])

    @task(retries=2, retry_delay_seconds=60)
    def opportunity_outcomes():
        return _run([sys.executable, '-m', 'AELP2.pipelines.opportunity_outcomes'])

    @task(retries=2, retry_delay_seconds=60)
    def portfolio_optimizer():
        return _run([sys.executable, '-m', 'AELP2.pipelines.portfolio_optimizer'])

    @task(retries=2, retry_delay_seconds=60)
    def bid_landscape_modeling():
        return _run([sys.executable, '-m', 'AELP2.pipelines.bid_landscape_modeling'])

    @task(retries=2, retry_delay_seconds=60)
    def dayparting_optimizer():
        return _run([sys.executable, '-m', 'AELP2.pipelines.dayparting_optimizer'])

    @task(retries=2, retry_delay_seconds=60)
    def competitive_intel_ingest():
        return _run([sys.executable, '-m', 'AELP2.pipelines.competitive_intel_ingest'])

    @task(retries=2, retry_delay_seconds=60)
    def security_audit():
        return _run([sys.executable, '-m', 'AELP2.pipelines.security_audit'])

    @task(retries=2, retry_delay_seconds=60)
    def quality_signal_daily():
        return _run([sys.executable, '-m', 'AELP2.pipelines.quality_signal_daily'])

    @task(retries=2, retry_delay_seconds=60)
    def policy_hints_writer():
        return _run([sys.executable, '-m', 'AELP2.pipelines.policy_hints_writer'])

    @task(retries=2, retry_delay_seconds=60)
    def offpolicy_eval():
        return _run([sys.executable, '-m', 'AELP2.pipelines.offpolicy_eval'])

    @task(retries=2, retry_delay_seconds=60)
    def creative_fatigue_alerts():
        return _run([sys.executable, '-m', 'AELP2.pipelines.creative_fatigue_alerts'])

    @task(retries=2, retry_delay_seconds=60)
    def realtime_budget_pacer():
        return _run([sys.executable, '-m', 'AELP2.pipelines.realtime_budget_pacer'])

    @task(retries=2, retry_delay_seconds=60)
    def rule_engine():
        return _run([sys.executable, '-m', 'AELP2.pipelines.rule_engine'])

    @task(retries=2, retry_delay_seconds=60)
    def ops_alerts_stub():
        return _run([sys.executable, '-m', 'AELP2.pipelines.ops_alerts_stub'])

    @task(retries=2, retry_delay_seconds=60)
    def audience_expansion():
        return _run([sys.executable, '-m', 'AELP2.pipelines.audience_expansion'])

    @task(retries=2, retry_delay_seconds=60)
    def budget_broker():
        return _run([sys.executable, '-m', 'AELP2.core.orchestration.budget_broker'])

    @task(retries=2, retry_delay_seconds=60)
    def canary_monitoring():
        return _run([sys.executable, '-m', 'AELP2.pipelines.canary_monitoring'])

    @task(retries=2, retry_delay_seconds=60)
    def trust_gates_evaluator():
        return _run([sys.executable, '-m', 'AELP2.pipelines.trust_gates_evaluator'])

    @task(retries=2, retry_delay_seconds=60)
    def permissions_check():
        return _run([sys.executable, '-m', 'AELP2.pipelines.permissions_check'])

    @task(retries=2, retry_delay_seconds=60)
    def canary_timeline_writer():
        return _run([sys.executable, '-m', 'AELP2.pipelines.canary_timeline_writer'])

    @task(retries=2, retry_delay_seconds=60)
    def creative_ab_planner():
        return _run([sys.executable, '-m', 'AELP2.pipelines.creative_ab_planner'])

    @task(retries=2, retry_delay_seconds=60)
    def copy_optimizer_stub():
        return _run([sys.executable, '-m', 'AELP2.pipelines.copy_optimizer_stub'])

    @task(retries=2, retry_delay_seconds=60)
    def lp_ab_hooks_stub():
        return _run([sys.executable, '-m', 'AELP2.pipelines.lp_ab_hooks_stub'])

    @task(retries=2, retry_delay_seconds=60)
    def policy_enforcer():
        return _run([sys.executable, '-m', 'AELP2.core.orchestration.policy_enforcer'])

    @task(retries=2, retry_delay_seconds=60)
    def cross_platform_kpi_daily():
        return _run([sys.executable, '-m', 'AELP2.pipelines.cross_platform_kpi_daily'])

    @task(retries=2, retry_delay_seconds=60)
    def bid_edit_proposals():
        return _run([sys.executable, '-m', 'AELP2.pipelines.bid_edit_proposals'])

    @task(retries=2, retry_delay_seconds=60)
    def hints_to_proposals():
        return _run([sys.executable, '-m', 'AELP2.pipelines.hints_to_proposals'])

    @task(retries=2, retry_delay_seconds=60)
    def parity_report():
        return _run([sys.executable, '-m', 'AELP2.pipelines.parity_report'])

    @task(retries=2, retry_delay_seconds=60)
    def bandit_v1():
        return _run([sys.executable, '-m', 'AELP2.core.optimization.bandit_service'])

    @task(retries=2, retry_delay_seconds=60)
    def budget_orchestrator():
        return _run([sys.executable, '-m', 'AELP2.core.optimization.budget_orchestrator'])

    @task(retries=2, retry_delay_seconds=60)
    def opportunity_scanner():
        return _run([sys.executable, '-m', 'AELP2.pipelines.opportunity_scanner', '--dry_run'])

    @task(retries=2, retry_delay_seconds=60)
    def channel_views():
        return _run([sys.executable, '-m', 'AELP2.pipelines.create_channel_views'])

    @task(retries=2, retry_delay_seconds=60)
    def recommendations_scanner():
        return _run([sys.executable, '-m', 'AELP2.pipelines.google_recommendations_scanner'])

    @task(retries=2, retry_delay_seconds=60)
    def upload_google_offline_conversions():
        return _run([sys.executable, '-m', 'AELP2.pipelines.upload_google_offline_conversions'])

    @task(retries=2, retry_delay_seconds=60)
    def upload_meta_capi_conversions():
        return _run([sys.executable, '-m', 'AELP2.pipelines.upload_meta_capi_conversions'])

    @flow(name="aelp2-nightly")
    def nightly_flow():
        import time
        t0 = time.time()
        # GX pre-checks (use dry-run if AELP2_GX_DRY_RUN=1)
        gx_cmd = [sys.executable, 'AELP2/ops/gx/run_checks.py']
        if os.getenv('AELP2_GX_DRY_RUN', '0') == '1':
            gx_cmd.append('--dry_run')
        rc0 = _run(gx_cmd)
        if rc0 != 0:
            # Gate: abort downstream tasks; log failure to ops tables
            try:
                from google.cloud import bigquery  # type: ignore
                project = os.getenv('GOOGLE_CLOUD_PROJECT')
                dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
                if project and dataset:
                    bq = bigquery.Client(project=project)
                    table_id = f"{project}.{dataset}.ops_flow_runs"
                    try:
                        bq.get_table(table_id)
                    except Exception:
                        schema = [
                            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
                            bigquery.SchemaField('flow', 'STRING'),
                            bigquery.SchemaField('rc_map', 'JSON'),
                            bigquery.SchemaField('failures', 'JSON'),
                            bigquery.SchemaField('ok', 'BOOL'),
                        ]
                        t = bigquery.Table(table_id, schema=schema)
                        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
                        bq.create_table(t)
                    rc_map = {'gx_checks': rc0}
                    bq.insert_rows_json(table_id, [{
                        'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                        'flow': 'aelp2-nightly',
                        'rc_map': json.dumps(rc_map),
                        'failures': json.dumps([('gx_checks', rc0)]),
                        'ok': False,
                    }])
                    # Alert
                    try:
                        table_alerts = f"{project}.{dataset}.ops_alerts"
                        try:
                            bq.get_table(table_alerts)
                        except Exception:
                            schema = [
                                bigquery.SchemaField('timestamp', 'TIMESTAMP'),
                                bigquery.SchemaField('alert', 'STRING'),
                                bigquery.SchemaField('severity', 'STRING'),
                                bigquery.SchemaField('details', 'JSON'),
                            ]
                            t = bigquery.Table(table_alerts, schema=schema)
                            t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
                            bq.create_table(t)
                        bq.insert_rows_json(table_alerts, [{
                            'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                            'alert': 'gx_gate_failed',
                            'severity': 'ERROR',
                            'details': json.dumps({'flow': 'aelp2-nightly'}),
                        }])
                    except Exception:
                        pass
            except Exception:
                pass
            return
        rc1 = mmm_v1()
        rc2 = bandit_v1()
        rc3 = budget_orchestrator()
        rc4 = opportunity_scanner()
        rc5 = channel_views()
        rc6 = recommendations_scanner()
        rc7 = upload_google_offline_conversions()
        rc8 = upload_meta_capi_conversions()
        # Simple alerting: non-zero return codes print a summary (hook to email/Slack later)
        failed = [(name, rc) for name, rc in [
            ('gx_checks', rc0),
            ('mmm_v1', rc1),
            ('bandit_v1', rc2),
            ('budget_orchestrator', rc3),
            ('opportunity_scanner', rc4),
            ('channel_views', rc5),
            ('recommendations_scanner', rc6),
            ('upload_google_offline_conversions', rc7),
            ('upload_meta_capi_conversions', rc8),
        ] if rc != 0]
        if failed:
            print(f"[flow] Failures: {failed}")
        # SLA check
        sla_sec = int(os.getenv('AELP2_FLOW_SLA_SEC', '0') or '0')
        sla_violation = False
        if sla_sec > 0:
            elapsed = time.time() - t0
            sla_violation = elapsed > sla_sec
        # Log to BigQuery ops_flow_runs if possible
        try:
            from google.cloud import bigquery  # type: ignore
            project = os.getenv('GOOGLE_CLOUD_PROJECT')
            dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
            if project and dataset:
                bq = bigquery.Client(project=project)
                table_id = f"{project}.{dataset}.ops_flow_runs"
                try:
                    bq.get_table(table_id)
                except Exception:
                    schema = [
                        bigquery.SchemaField('timestamp', 'TIMESTAMP'),
                        bigquery.SchemaField('flow', 'STRING'),
                        bigquery.SchemaField('rc_map', 'JSON'),
                        bigquery.SchemaField('failures', 'JSON'),
                        bigquery.SchemaField('ok', 'BOOL'),
                    ]
                    t = bigquery.Table(table_id, schema=schema)
                    t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
                    bq.create_table(t)
                rc_map = {
                    'gx_checks': rc0,
                    'mmm_v1': rc1,
                    'bandit_v1': rc2,
                    'budget_orchestrator': rc3,
                    'opportunity_scanner': rc4,
                    'channel_views': rc5,
                    'recommendations_scanner': rc6,
                    'upload_google_offline_conversions': rc7,
                    'upload_meta_capi_conversions': rc8,
                }
                row = {
                    'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                    'flow': 'aelp2-nightly',
                    'rc_map': json.dumps(rc_map),
                    'failures': json.dumps(failed),
                    'ok': len(failed) == 0 and not sla_violation,
                }
                bq.insert_rows_json(table_id, [row])
                # Write ops_alert if SLA violated
                if sla_violation:
                    try:
                        table_alerts = f"{project}.{dataset}.ops_alerts"
                        try:
                            bq.get_table(table_alerts)
                        except Exception:
                            schema = [
                                bigquery.SchemaField('timestamp', 'TIMESTAMP'),
                                bigquery.SchemaField('alert', 'STRING'),
                                bigquery.SchemaField('severity', 'STRING'),
                                bigquery.SchemaField('details', 'JSON'),
                            ]
                            t = bigquery.Table(table_alerts, schema=schema)
                            t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
                            bq.create_table(t)
                        bq.insert_rows_json(table_alerts, [{
                            'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                            'alert': 'flow_sla_violation',
                            'severity': 'WARNING',
                            'details': json.dumps({'flow': 'aelp2-nightly', 'sla_sec': sla_sec}),
                        }])
                    except Exception:
                        pass
        except Exception as e:
            print(f"[flow] BQ logging failed: {e}")
        # Optional Slack webhook alert on failures
        try:
            if failed and os.getenv('AELP2_SLACK_WEBHOOK_URL'):
                import urllib.request
                data = json.dumps({'text': f"AELP2 nightly flow failures: {failed}"}).encode('utf-8')
                req = urllib.request.Request(os.environ['AELP2_SLACK_WEBHOOK_URL'], data=data, headers={'Content-Type': 'application/json'})
                urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[flow] Slack alert failed: {e}")

    # Optional simple weekly flow focused on MMM
    @flow(name="aelp2-weekly-mmm")
    def weekly_mmm_flow():
        rc1 = mmm_lightweightmmm(); rc2 = robyn_validator(); rc3 = channel_attribution_weekly(); rc4 = journeys_populate()
        rc5 = uplift_eval(); rc6 = propensity_scores(); rc7 = segments_to_audiences(); rc8 = training_posthoc()
        rc9 = youtube_reach_planner(); rc10 = kpi_consistency_check(); rc11 = opportunity_outcomes(); rc12 = portfolio_optimizer()
        rc13 = bid_landscape_modeling(); rc14 = dayparting_optimizer(); rc15 = competitive_intel_ingest(); rc16 = security_audit()
        rc17 = quality_signal_daily(); rc18 = policy_hints_writer(); rc19 = offpolicy_eval(); rc20 = creative_fatigue_alerts()
        rc21 = realtime_budget_pacer(); rc22 = rule_engine(); rc23 = ops_alerts_stub(); rc24 = audience_expansion()
        rc25 = budget_broker(); rc26 = canary_monitoring(); rc27 = trust_gates_evaluator(); rc28 = permissions_check()
        rc29 = canary_timeline_writer(); rc30 = creative_ab_planner(); rc31 = copy_optimizer_stub(); rc32 = lp_ab_hooks_stub()
        rc33 = policy_enforcer(); rc34 = cross_platform_kpi_daily(); rc35 = bid_edit_proposals(); rc36 = hints_to_proposals(); rc37 = parity_report()
        # Log to ops_flow_runs
        try:
            from google.cloud import bigquery  # type: ignore
            project = os.getenv('GOOGLE_CLOUD_PROJECT')
            dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
            if project and dataset:
                bq = bigquery.Client(project=project)
                table_id = f"{project}.{dataset}.ops_flow_runs"
                try:
                    bq.get_table(table_id)
                except Exception:
                    schema = [
                        bigquery.SchemaField('timestamp', 'TIMESTAMP'),
                        bigquery.SchemaField('flow', 'STRING'),
                        bigquery.SchemaField('rc_map', 'JSON'),
                        bigquery.SchemaField('failures', 'JSON'),
                        bigquery.SchemaField('ok', 'BOOL'),
                    ]
                    t = bigquery.Table(table_id, schema=schema)
                    t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
                    bq.create_table(t)
                rc_map = {
                    'mmm_lightweightmmm': rc1,
                    'robyn_validator': rc2,
                    'channel_attribution_weekly': rc3,
                    'journeys_populate': rc4,
                    'uplift_eval': rc5,
                    'propensity_scores': rc6,
                    'segments_to_audiences': rc7,
                    'training_posthoc': rc8,
                    'youtube_reach_planner': rc9,
                    'kpi_consistency_check': rc10,
                    'opportunity_outcomes': rc11,
                    'portfolio_optimizer': rc12,
                    'bid_landscape_modeling': rc13,
                    'dayparting_optimizer': rc14,
                    'competitive_intel_ingest': rc15,
                    'security_audit': rc16,
                    'quality_signal_daily': rc17,
                    'policy_hints_writer': rc18,
                    'offpolicy_eval': rc19,
                    'creative_fatigue_alerts': rc20,
                    'realtime_budget_pacer': rc21,
                    'rule_engine': rc22,
                    'ops_alerts_stub': rc23,
                    'audience_expansion': rc24,
                    'budget_broker': rc25,
                    'canary_monitoring': rc26,
                    'trust_gates_evaluator': rc27,
                    'permissions_check': rc28,
                    'canary_timeline_writer': rc29,
                    'creative_ab_planner': rc30,
                    'copy_optimizer_stub': rc31,
                    'lp_ab_hooks_stub': rc32,
                    'policy_enforcer': rc33,
                    'cross_platform_kpi_daily': rc34,
                    'bid_edit_proposals': rc35,
                    'hints_to_proposals': rc36,
                    'parity_report': rc37,
                }
                failed = [(k, v) for k, v in rc_map.items() if v != 0]
                row = {
                    'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                    'flow': 'aelp2-weekly-mmm',
                    'rc_map': json.dumps(rc_map),
                    'failures': json.dumps(failed),
                    'ok': len(failed) == 0,
                }
                bq.insert_rows_json(table_id, [row])
                # Optional Slack on failures
                try:
                    if failed and os.getenv('AELP2_SLACK_WEBHOOK_URL'):
                        import urllib.request
                        data = json.dumps({'text': f"AELP2 weekly flow failures: {failed}"}).encode('utf-8')
                        req = urllib.request.Request(os.environ['AELP2_SLACK_WEBHOOK_URL'], data=data, headers={'Content-Type': 'application/json'})
                        urllib.request.urlopen(req, timeout=5)
                except Exception as e:
                    print(f"[flow] Slack alert failed: {e}")
        except Exception as e:
            print(f"[flow] BQ logging failed: {e}")

    if __name__ == '__main__':
        nightly_flow()
        # Users can also run: `python3 -m AELP2.ops.prefect_flows weekly_mmm`
        # but we default to nightly when invoked as a module.
else:
    if __name__ == '__main__':
        print('Prefect not installed. To use flows: pip install prefect; then run this module again.')
