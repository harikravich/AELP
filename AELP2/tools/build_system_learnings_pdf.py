#!/usr/bin/env python3
from __future__ import annotations
"""
Build a narrative, data‑rich PDF for business stakeholders capturing:
 - Core problem and why: simulate real life to train/validate policies offline
 - How AELP↔AELP2 connect; architecture of RL simulator and data pipelines
 - What we learned (design choices, what works, risks)
 - First‑wave outputs (slate, budgets, CAC/signups, accuracy), with ad thumbnails

Writes:
  - AELP2/docs/AELP2_System_Learnings.pdf
  - AELP2/docs/AELP2_System_Learnings.html (quick browse)
"""
import os, json, datetime, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'AELP2' / 'docs'
REPORTS = ROOT / 'AELP2' / 'reports'
FINALS = ROOT / 'AELP2' / 'outputs' / 'finals'

def require(mod: str):
    try:
        return __import__(mod)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', mod])
        return __import__(mod)

rl = require('reportlab')
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch

def ascii_arch_connect() -> str:
    return textwrap.dedent('''
        AELP  (foundational experiments, legacy pipelines)
           \
            \
             v
        AELP2 (production pipelines + planner + RL sim)
             |
             |  Meta Insights   Vendor/Ad Library
             v        \         /
        +-----------------------------+
        |  Ingestion + Normalization  |
        +-----------------------------+
             |
             v
        +-----------------------------+
        | Features + Ranker Scoring   |
        | (new‑ad p_win, lcb)         |
        +-----------------------------+
             |
             v
        +-----------------------------+      +-------------------+
        | Baselines (by placement)    | ---> | Monte‑Carlo       |
        +-----------------------------+      | CAC/Volume        |
             |                               +-------------------+
             v                                         |
        +-----------------------------+                 v
        | Offline RL Simulator        | ------->  Portfolio policies
        +-----------------------------+                 |
             |                                       Planner UI/API
             v                                         |
        +-----------------------------+                 v
        | Launch Playbooks (Meta)     | -------->  Live tests & feedback
        +-----------------------------+
    ''')

def ascii_rl_sim() -> str:
    return textwrap.dedent('''
        RL Simulator (offline)
        ---------------------
        Inputs:
         - Per‑blueprint priors: p_win, novelty
         - Baseline draws: CPM, CTR, CVR by placement (triangular p10/p50/p90)
         - Budget constraints & caps

        Loop:
         1) Sample CTR/CVR/CPM → estimate signups/impressions and CAC
         2) Thompson‑sample success rate per arm (signups/imps proxy)
         3) Allocate daily budget to top‑sampled arms (with caps)
         4) Update posteriors using simulated outcomes
         5) Emit daily allocations + expected CAC/signups

        Outputs:
         - rl_offline_simulation.json (ranking, history)
         - Portfolio recommendation (8–12 arms + early‑stop rules)
    ''')

def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def top_portfolio(fore_path: Path, budget: int = 30000, k: int = 8):
    data = read_json(fore_path, {}) or {}
    items = sorted(data.get('items', []), key=lambda r: -float(r.get('p_win',0)))[:k]
    per = budget/float(k)
    tot = {'p10':0.0,'p50':0.0,'p90':0.0}
    rows=[]
    for r in items:
        fx = r.get('budgets',{}).get(str(budget),{})
        sc = per/float(budget)
        su = {q: float((fx.get('signups') or {}).get(q,0.0))*sc for q in ('p10','p50','p90')}
        rows.append({'id': r['creative_id'], 'p_win': float(r.get('p_win',0)), 'su_p50': su['p50'], 'cac_p50': float((fx.get('cac') or {}).get('p50',0.0))})
        for q in ('p10','p50','p90'): tot[q]+=su[q]
    return {'rows': rows, 'tot': tot, 'budget': budget, 'k': k}

def pick_thumbs() -> list[Path]:
    thumbs=[]
    if FINALS.exists():
        for name in ('cool_preview_v1.jpg','orig_freeze_the_chaos_2.jpg','orig_spot_the_scam_1.jpg'):
            p = FINALS / name
            if p.exists(): thumbs.append(p)
    return thumbs

def build_pdf():
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / 'AELP2_System_Learnings.pdf'
    doc = SimpleDocTemplate(str(out), pagesize=LETTER, leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)

    styles = getSampleStyleSheet()
    body = ParagraphStyle('Body', parent=styles['BodyText'], leading=14, fontName='Helvetica')
    code = ParagraphStyle('Code', parent=styles['BodyText'], fontName='Courier', leading=12)
    h1 = ParagraphStyle('H1', parent=styles['Heading1'], spaceAfter=6)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceAfter=4)

    story=[]
    story += [Paragraph('AELP / AELP2 — System Learnings & First‑Wave Outputs', h1),
              Paragraph(datetime.datetime.utcnow().strftime('Generated %Y-%m-%d %H:%M UTC'), styles['Normal']),
              Spacer(1,0.2*inch)]

    # Problem & Why
    story += [Paragraph('1) Core problem and why', h2),
              Paragraph('We need to simulate real‑life response to creatives and placements so the RL agent can learn policies without burning budget. '
                        'AELP2 turns historical Meta performance and vendor creative evidence into placement‑aware baselines and calibrated forecasts, '
                        'then uses an offline bandit simulator to learn allocation policies and output a slate and launch playbook.', body)]

    # Architecture diagrams
    story += [Paragraph('2) Architecture — AELP↔AELP2 and RL simulator', h2),
              Paragraph('<pre>%s</pre>' % ascii_arch_connect().replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'), code),
              Paragraph('<pre>%s</pre>' % ascii_rl_sim().replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'), code)]

    # Pipelines summary
    story += [Paragraph('3) Data pipelines and system flow', h2),
              Paragraph('• Meta→BQ by placement with backoff/window slicing.\n'
                        '• Vendor/Ad Library via SearchAPI or CSV; normalized into creative_objects.\n'
                        '• Features + new‑ad ranker scoring (p_win + conformal lcb).\n'
                        '• Baselines (CPM/CTR/CVR) from US paid data; MC forecasts for CAC/volume.\n'
                        '• Offline RL sim (Thompson sampling on signups/imps proxy) to propose a portfolio.\n'
                        '• Planner API/UI packages $30k/$50k with playbooks.', body)]

    # Learnings
    story += [Paragraph('4) What we learned (why this design)', h2),
              Paragraph('• Placement‑aware baselines reduce drift; CVR clamps avoid outliers.\n'
                        '• Scoring with p_win helps prioritize motifs; conformal lower bounds provide caution.\n'
                        '• Offline RL accelerates exploration safely (policies pre‑tested); early‑stop rules protect downside.\n'
                        '• US Ad Library has gaps; augment with SearchAPI/vendor CSV; rely on our own BQ for performance priors.', body)]

    # Forecast & slate
    sec = top_portfolio(REPORTS/'us_cac_volume_forecasts.json', 30000, 8)
    bal = top_portfolio(REPORTS/'us_balance_forecasts.json', 30000, 8)
    AOV_SEC=200.0; AOV_BAL=120.0
    sec_rev = sec['tot']['p50']*AOV_SEC
    bal_rev = bal['tot']['p50']*AOV_BAL
    comb_su = sec['tot']['p50']+bal['tot']['p50']
    comb_rev = sec_rev+bal_rev
    comb_spend = 60000.0
    comb_cac = comb_spend/max(comb_su,1e-6)

    story += [Paragraph('5) First‑wave outputs (recommended slate)', h2),
              Paragraph(f"Security $30k/day → signups p50: {sec['tot']['p50']:.1f}, CAC p50≈${comb_spend/ (sec['tot']['p50']+bal['tot']['p50']):.1f} combined", body),
              Paragraph(f"Balance $30k/day → signups p50: {bal['tot']['p50']:.1f}", body),
              Paragraph(f"Combined $60k/day → signups p50: {comb_su:.1f}, CAC p50≈${comb_cac:.1f}, revenue p50≈${comb_rev:,.0f}", body)]

    story += [Paragraph('Security top 8 (id, p_win, su_p50, CAC p50)', h2)]
    for r in sec['rows']:
        story += [Paragraph(f"• {r['id']} — p_win {r['p_win']:.3f}; su_p50 {r['su_p50']:.1f}; CAC ${r['cac_p50']:.0f}", body)]
    story += [Paragraph('Balance top 8 (id, p_win, su_p50, CAC p50)', h2)]
    for r in bal['rows']:
        story += [Paragraph(f"• {r['id']} — p_win {r['p_win']:.3f}; su_p50 {r['su_p50']:.1f}; CAC ${r['cac_p50']:.0f}", body)]

    # Accuracy snapshot
    acc = read_json(REPORTS/'ad_level_accuracy_v23.json', {}) or {}
    story += [Paragraph('6) Accuracy & coverage', h2),
              Paragraph(f"Ranker precision@5: {acc.get('precision_at_5')}, precision@10: {acc.get('precision_at_10')}", body)]
    calib = REPORTS/'ad_calibration_reliability.png'
    if calib.exists():
        story += [Image(str(calib), width=6.5*inch, height=3.2*inch)]

    # Thumbnails of ads we have
    thumbs = pick_thumbs()
    if thumbs:
        story += [Paragraph('7) Example creatives (thumbnails)', h2)]
        for img in thumbs:
            story += [Paragraph(img.name, body), Image(str(img), width=3.2*inch, height=1.8*inch), Spacer(1,0.1*inch)]

    # Next steps
    story += [Paragraph('8) Next steps', h2),
              Paragraph('• Complete 90‑day by‑placement backfill and recalibrate.\n'
                        '• Expand Balance set and run offline RL across offer variants.\n'
                        '• Launch controlled live test with early‑stop rules and measure lift vs. historical CAC.', body)]

    doc.build(story)
    return out

def build_html(pdf_out: Path):
    html = DOCS / 'AELP2_System_Learnings.html'
    lines = ["<html><head><meta charset='utf-8'><style>body{font-family:system-ui,Arial;margin:24px;line-height:1.5} pre{background:#0f1115;color:#eaeaea;padding:12px;border-radius:8px;}</style><title>AELP2 Learnings</title></head><body>"]
    lines.append('<h1>AELP / AELP2 — System Learnings & First‑Wave Outputs</h1>')
    lines.append(f"<div>Generated {datetime.datetime.utcnow().isoformat()}Z</div>")
    lines.append('<h2>Architecture</h2>')
    lines.append('<pre>'+ascii_arch_connect()+'</pre>')
    lines.append('<pre>'+ascii_rl_sim()+'</pre>')
    lines.append('<p>See PDF for detailed numbers and thumbnails.</p>')
    lines.append('</body></html>')
    html.write_text('\n'.join(lines))
    return html

def main():
    pdf = build_pdf()
    html = build_html(pdf)
    print(json.dumps({'pdf': str(pdf), 'html': str(html)}, indent=2))

if __name__ == '__main__':
    main()

