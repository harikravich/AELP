#!/usr/bin/env python3
"""
Generate a Macro Account Audit checklist for Meta (offline).

Reads recent simulator fidelity (if present) to suggest priority areas.
Writes AELP2/reports/meta_account_audit.md.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'AELP2' / 'reports'

def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def main():
    fid = load(REPORTS / 'sim_fidelity_campaigns_temporal_v2.json')
    mape_p = (fid.get('summary') or {}).get('purchases_day_mape')
    mape_c = (fid.get('summary') or {}).get('cac_day_mape')
    cov = (fid.get('summary') or {}).get('coverage80')

    lines = []
    lines.append('# Meta Macro Account Audit (Offline)')
    lines.append('')
    lines.append('Status')
    lines.append(f'- Simulator fidelity (last window): Purch MAPE={mape_p}%, CAC MAPE={mape_c}%, 80% PI coverage={cov}%')
    lines.append('')
    lines.append('## Measurement & Signals')
    lines.append('- [ ] Pixel + CAPI Gateway configured; no duplicate events (server vs browser)')
    lines.append('- [ ] Primary conversion optimized in Ads (Purchases / Complete Registration as needed)')
    lines.append('- [ ] Attribution window aligned to business goals (e.g., 7-day click, 1-day view)')
    lines.append('- [ ] AEM priorities set; shopping signals mapped where applicable')
    lines.append('')
    lines.append('## Structure & Budgets')
    lines.append('- [ ] Campaign structure: CBO for mature groups; ABO for tests')
    lines.append('- [ ] Test budget cap ≤ 10% of daily spend (when live)')
    lines.append('- [ ] Cost caps on stabilized ad sets; guard CAC at campaign level')
    lines.append('')
    lines.append('## Placements & Coverage')
    lines.append('- [ ] Placements include Reels/Stories; creative ratios 1:1, 4:5, 9:16 covered')
    lines.append('- [ ] Captions on all video; legibility contrast ≥ 4.5:1')
    lines.append('')
    lines.append('## Creative & Brand Safety')
    lines.append('- [ ] Approved template IDs only; logo safe-zones enforced')
    lines.append('- [ ] COPPA/FTC compliance (parent-focused for Balance; truthful, substantiated claims)')
    lines.append('- [ ] Audio loudness ≈ -14 LUFS; watermark policy if testing')
    lines.append('')
    lines.append('## Launch Readiness (when live)')
    lines.append('- [ ] CAC guardrails configured; rollback triggers tested')
    lines.append('- [ ] UTM conventions in place; landing pages performance-tested')
    lines.append('- [ ] Reporting: daily diff, creative leaderboard, simulator fidelity panel')
    lines.append('')
    lines.append('Notes')
    lines.append('- Add specific remediation items here after account review.')

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'meta_account_audit.md').write_text('\n'.join(lines))
    print('Wrote AELP2/reports/meta_account_audit.md')

if __name__ == '__main__':
    main()

