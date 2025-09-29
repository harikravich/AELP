"""
Subagent Orchestrator (Skeleton)

Coordinates parallel subagents in a safe, observable way. Intended to be called
periodically from the main training loop when enabled via env flags.

Flags:
- AELP2_SUBAGENTS_ENABLE=1
- AELP2_SUBAGENTS_LIST="creative,budget" (comma-separated)
- AELP2_SUBAGENTS_SHADOW=1 (default)
- AELP2_SUBAGENTS_CADENCE_STEPS=10

Per-subagent enables:
- AELP2_ENABLE_SUBAGENT_CREATIVE=1
- AELP2_ENABLE_SUBAGENT_BUDGET=1
"""

import os
import logging
from typing import Dict, Any, List, Optional

from AELP2.core.safety.hitl import validate_action_safety
from AELP2.core.monitoring.bq_writer import create_bigquery_writer, BigQueryWriter

logger = logging.getLogger(__name__)


class BaseSubagent:
    def name(self) -> str:
        return self.__class__.__name__

    def is_enabled(self) -> bool:
        return True

    def run_step(self, state: Any) -> Optional[Dict[str, Any]]:
        return None

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok"}


class CreativeIdeationSubagent(BaseSubagent):
    def is_enabled(self) -> bool:
        return os.getenv('AELP2_ENABLE_SUBAGENT_CREATIVE', '0') == '1'

    def run_step(self, state: Any) -> Optional[Dict[str, Any]]:
        # Skeleton proposal example; real implementation would score ideas
        return {
            'type': 'creative_change',
            'creative': {'headline': 'Try Aura Premium', 'description': 'Protect your family today'},
            'channel': 'search'
        }


class BudgetRebalancerSubagent(BaseSubagent):
    def is_enabled(self) -> bool:
        return os.getenv('AELP2_ENABLE_SUBAGENT_BUDGET', '0') == '1'

    def _get_bq_client(self):
        from google.cloud import bigquery
        project = os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not project:
            raise RuntimeError('GOOGLE_CLOUD_PROJECT not set')
        return bigquery.Client(project=project)

    def _find_candidates(self, lookback_days: int = 14) -> Optional[Dict[str, Any]]:
        try:
            client = self._get_bq_client()
            project = os.environ['GOOGLE_CLOUD_PROJECT']
            dataset = os.environ['BIGQUERY_TRAINING_DATASET']
            sql = f"""
                WITH agg AS (
                  SELECT
                    campaign_id,
                    SUM(cost_micros)/1e6 AS cost,
                    SUM(conversions) AS conv,
                    SUM(conversion_value) AS revenue
                  FROM `{project}.{dataset}.ads_campaign_performance`
                  WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY) AND CURRENT_DATE()
                  GROUP BY campaign_id
                )
                SELECT campaign_id,
                       cost,
                       conv,
                       revenue,
                       SAFE_DIVIDE(revenue, NULLIF(cost,0)) AS roas
                FROM agg
                HAVING cost > 0
            """
            rows = list(client.query(sql).result())
            if not rows:
                return None
            # Convert to list of dicts
            rows = [dict(r) for r in rows]
            # Sort by ROAS desc for winners and asc for laggards
            winners = sorted(rows, key=lambda r: (r['roas'] or 0.0), reverse=True)
            laggards = sorted(rows, key=lambda r: (r['roas'] or 0.0))
            return {
                'best': winners[0] if winners else None,
                'worst': laggards[0] if laggards else None
            }
        except Exception:
            return None

    def run_step(self, state: Any) -> Optional[Dict[str, Any]]:
        # Guardrail params
        try:
            max_delta_pct = float(os.getenv('AELP2_BUDGET_REBALANCER_MAX_DELTA_PCT', '0.1'))
        except Exception:
            max_delta_pct = 0.1
        max_abs_delta = float(os.getenv('AELP2_BUDGET_REBALANCER_MAX_ABS_DELTA', '0'))  # 0 disables
        lookback_days = int(os.getenv('AELP2_BUDGET_REBALANCER_LOOKBACK_DAYS', '14'))
        pacer_awareness = os.getenv('AELP2_BUDGET_REBALANCER_PACER_AWARE', '1') == '1'
        pacer_cut_factor = float(os.getenv('AELP2_BUDGET_REBALANCER_PACER_CUT', '0.5'))

        # Query candidates
        picks = self._find_candidates(lookback_days=lookback_days)
        if not picks or (not picks.get('best') and not picks.get('worst')):
            return None

        proposals: List[Dict[str, Any]] = []
        best = picks.get('best')
        worst = picks.get('worst')

        def _apply_caps(delta_pct: float, metrics: Dict[str, Any]) -> Dict[str, Any]:
            # Pacer awareness: if cost is high but conversions/roas weak, cut increases
            if pacer_awareness and metrics:
                roas = float(metrics.get('roas') or 0.0)
                cost = float(metrics.get('cost') or 0.0)
                if roas < 1.0 and cost > 100.0:  # heuristic guard
                    delta_pct = max(0.0, delta_pct * pacer_cut_factor)
            # Absolute cap: convert to abs if budget context present later
            return {
                'delta_pct': delta_pct,
                'max_abs_delta': max_abs_delta if max_abs_delta > 0 else None,
            }

        if best and best.get('roas') and best['roas'] > 1.0:
            caps = _apply_caps(max_delta_pct, {'roas': best['roas'], 'cost': best['cost']})
            proposals.append({
                'type': 'budget_increase',
                'campaign_id': str(best['campaign_id']),
                'delta_pct': caps['delta_pct'],
                'max_abs_delta': caps['max_abs_delta'],
                'basis': 'roas_winner',
                'metrics': {'roas': float(best['roas']), 'cost': float(best['cost']), 'revenue': float(best['revenue'])}
            })
        if worst and (worst.get('roas') is not None) and worst['roas'] < 0.8:
            dec_pct = max_delta_pct / 2.0
            caps = _apply_caps(dec_pct, {'roas': worst['roas'], 'cost': worst['cost']})
            proposals.append({
                'type': 'budget_decrease',
                'campaign_id': str(worst['campaign_id']),
                'delta_pct': caps['delta_pct'],
                'max_abs_delta': caps['max_abs_delta'],
                'basis': 'roas_laggard',
                'metrics': {'roas': float(worst['roas']), 'cost': float(worst['cost']), 'revenue': float(worst['revenue'])}
            })

        # Return a single proposal per tick to avoid flooding
        if proposals:
            return proposals[0]
        return None


class DriftMonitorSubagent(BaseSubagent):
    """Detects drift/underperformance and proposes recalibration or guard tweaks."""

    def is_enabled(self) -> bool:
        return os.getenv('AELP2_ENABLE_SUBAGENT_DRIFT', '0') == '1'

    def run_step(self, state: Any) -> Optional[Dict[str, Any]]:
        # Expect caller to pass metrics into orchestrator.run_once(...)
        # Here we use env for thresholds if needed.
        return None  # Logic handled in orchestrator via provided metrics

    def analyze(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Check win-rate severe underperformance
        try:
            min_win_rate = float(os.getenv('AELP2_MIN_WIN_RATE', '0.15'))
        except Exception:
            min_win_rate = 0.15
        wr = float(metrics.get('win_rate', 0.0))
        if wr < min_win_rate * 0.5:
            return {
                'type': 'recalibrate_auctions',
                'reason': 'win_rate_below_half_threshold',
                'details': {'win_rate': wr, 'min_win_rate': min_win_rate}
            }
        # Check last fidelity evaluation for failures
        try:
            from google.cloud import bigquery
            project = os.environ['GOOGLE_CLOUD_PROJECT']
            dataset = os.environ['BIGQUERY_TRAINING_DATASET']
            client = bigquery.Client(project=project)
            sql = f"""
                SELECT passed, mape_roas, rmse_roas, mape_cac, rmse_cac, ks_winrate_vs_impressionshare
                FROM `{project}.{dataset}.fidelity_evaluations`
                ORDER BY timestamp DESC
                LIMIT 1
            """
            rows = list(client.query(sql).result())
            if rows:
                r = dict(rows[0])
                if not r.get('passed', True):
                    return {
                        'type': 'recalibrate_auctions',
                        'reason': 'fidelity_eval_failed',
                        'details': r
                    }
        except Exception:
            pass
        return None


class TargetingDiscoverySubagent(BaseSubagent):
    """Suggest negative keywords from search terms with cost but zero conversions."""

    def is_enabled(self) -> bool:
        return os.getenv('AELP2_ENABLE_SUBAGENT_TARGETING', '0') == '1'

    def _get_bq_client(self):
        from google.cloud import bigquery
        project = os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not project:
            raise RuntimeError('GOOGLE_CLOUD_PROJECT not set')
        return bigquery.Client(project=project)

    def run_step(self, state: Any) -> Optional[Dict[str, Any]]:
        try:
            client = self._get_bq_client()
            project = os.environ['GOOGLE_CLOUD_PROJECT']
            dataset = os.environ['BIGQUERY_TRAINING_DATASET']
            min_cost = float(os.getenv('AELP2_TARGETING_MIN_COST', '20'))
            sql = f"""
                SELECT customer_id, campaign_id, ad_group_id,
                       ANY_VALUE(search_term_hash) AS term_hash,
                       SUM(cost_micros)/1e6 AS cost,
                       SUM(conversions) AS conv
                FROM `{project}.{dataset}.ads_search_terms`
                WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
                GROUP BY customer_id, campaign_id, ad_group_id
                HAVING conv = 0 AND cost >= {min_cost}
                LIMIT 1
            """
            rows = list(client.query(sql).result())
            if not rows:
                return None
            r = dict(rows[0])
            return {
                'type': 'targeting_change',
                'action': 'add_negative_keyword',
                'customer_id': r['customer_id'],
                'campaign_id': str(r['campaign_id']),
                'ad_group_id': str(r['ad_group_id']),
                'term_hash': r['term_hash'],
                'metrics': {'cost': float(r['cost']), 'conversions': float(r['conv'])}
            }
        except Exception:
            return None


class AttributionDiagnosticsSubagent(BaseSubagent):
    def is_enabled(self) -> bool:
        return os.getenv('AELP2_ENABLE_SUBAGENT_ATTRIBUTION', '0') == '1'

    def _get_bq_client(self):
        from google.cloud import bigquery
        project = os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not project:
            raise RuntimeError('GOOGLE_CLOUD_PROJECT not set')
        return bigquery.Client(project=project)

    def run_step(self, state: Any) -> Optional[Dict[str, Any]]:
        # Shadow-only diagnostic: compare Ads CAC vs GA4 conversions trend and surface if divergent
        try:
            client = self._get_bq_client()
            project = os.environ['GOOGLE_CLOUD_PROJECT']
            dataset = os.environ['BIGQUERY_TRAINING_DATASET']
            sql = f"""
              WITH ads AS (
                SELECT DATE(date) AS d, SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac
                FROM `{project}.{dataset}.ads_campaign_performance`
                WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
                GROUP BY d
              ), ga4 AS (
                SELECT DATE(date) AS d, SUM(conversions) AS conv
                FROM `{project}.{dataset}.ga4_daily`
                WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
                GROUP BY d
              )
              SELECT AVG(a.cac) AS avg_cac, AVG(g.conv) AS avg_conv
              FROM ads a JOIN ga4 g USING(d)
            """
            rows = list(client.query(sql).result())
            if not rows:
                return None
            r = dict(rows[0])
            return {
                'type': 'attribution_diagnostics',
                'action': 'review_kpi_alignment',
                'metrics': {'avg_cac': float(r.get('avg_cac') or 0.0), 'avg_conv': float(r.get('avg_conv') or 0.0)},
            }
        except Exception:
            return None


def _load_subagents(list_csv: str) -> List[BaseSubagent]:
    mapping = {
        'creative': CreativeIdeationSubagent,
        'budget': BudgetRebalancerSubagent,
        'drift': DriftMonitorSubagent,
        'targeting': TargetingDiscoverySubagent,
        'attribution': AttributionDiagnosticsSubagent,
    }
    out: List[BaseSubagent] = []
    for name in [s.strip() for s in list_csv.split(',') if s.strip()]:
        cls = mapping.get(name)
        if not cls:
            logger.warning(f"Unknown subagent '{name}', skipping")
            continue
        agent = cls()
        if agent.is_enabled():
            out.append(agent)
    return out


class SubagentOrchestrator:
    def __init__(self):
        self.enabled = os.getenv('AELP2_SUBAGENTS_ENABLE', '0') == '1'
        self.shadow = os.getenv('AELP2_SUBAGENTS_SHADOW', '1') == '1'
        self.cadence = int(os.getenv('AELP2_SUBAGENTS_CADENCE_STEPS', '10'))
        self.subagents = _load_subagents(os.getenv('AELP2_SUBAGENTS_LIST', '')) if self.enabled else []
        self._bq: Optional[BigQueryWriter] = None
        # Quotas and rate limits
        self._max_per_episode = int(os.getenv('AELP2_SUBAGENTS_MAX_PROPOSALS_PER_EPISODE', '10'))
        self._proposals_this_episode = 0
        try:
            self._bq = create_bigquery_writer()
        except Exception as e:
            logger.warning(f"BigQuery writer not available for subagents: {e}")

    def run_once(self, state: Any, episode_id: Optional[str], step: int, metrics: Dict[str, Any]):
        if not self.enabled:
            return
        if self.cadence > 1 and (step % self.cadence) != 0:
            return
        if self._proposals_this_episode >= self._max_per_episode:
            return
        for agent in self.subagents:
            try:
                proposal = agent.run_step(state)
                # Allow drift monitor to analyze metrics
                if not proposal and isinstance(agent, DriftMonitorSubagent):
                    proposal = agent.analyze(metrics)
                if not proposal:
                    continue
                # Safety/HITL
                is_safe, violations, approval_id = validate_action_safety(
                    action=proposal,
                    metrics=metrics,
                    context={'episode_id': episode_id, 'step': step, 'risk_level': 'medium'}
                )
                event_meta = {
                    'proposal': proposal,
                    'violations': violations,
                    'approval_id': approval_id,
                    'shadow': self.shadow,
                }
                if self._bq:
                    self._bq.write_subagent_event({
                        'subagent': agent.name(),
                        'event_type': 'proposal',
                        'status': 'safe' if is_safe else 'violations',
                        'episode_id': episode_id,
                        'metadata': event_meta,
                    })
                # Shadow-only in this skeleton
                if not self.shadow:
                    # Hook real platform adapter calls behind strict flags (not implemented here)
                    pass
                self._proposals_this_episode += 1
            except Exception as e:
                logger.error(f"Subagent {agent.name()} failed: {e}")
                if self._bq:
                    self._bq.write_subagent_event({
                        'subagent': agent.name(),
                        'event_type': 'error',
                        'status': 'exception',
                        'episode_id': episode_id,
                        'metadata': {'error': str(e)}
                    })
    def reset_episode(self):
        """Reset per-episode quotas/counters."""
        self._proposals_this_episode = 0
