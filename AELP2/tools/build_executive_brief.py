#!/usr/bin/env python3
from __future__ import annotations
"""
Executive Brief (business-first, no code/jargon):
 - Problem framing, goals, and constraints
 - Our solution and why it works
 - Simple architecture and flow
 - What we built (in plain language)
 - First-wave outputs (slate, budgets, CAC, volume, revenue)
 - Confidence & accuracy
 - Risks & mitigations
 - What to do next (90-day plan)

Writes: AELP2/docs/Executive_Brief_AELP2.pdf (+ HTML)
"""
import json, datetime, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'AELP2' / 'docs'
REPORTS = ROOT / 'AELP2' / 'reports'

def require(mod:str):
    try:
        return __import__(mod)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable,'-m','pip','install',mod])
        return __import__(mod)

rl = require('reportlab')
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch

def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def ascii_arch():
    return textwrap.dedent('''
        How It Works (Simple View)
        -------------------------
        1) We load past performance and competitor signals.
        2) We estimate realistic costs and conversion rates by placement.
        3) We score creative concepts (which ones likely win).
        4) We simulate outcomes offline and pick a daily slate + budgets.
        5) We hand you a launch playbook and track results.

        [Data] → [Baselines] → [Scoring] → [Simulation] → [Planner] → [Launch]
    ''')

def top_portfolio(path: Path, budget=30000, k=8):
    j = read_json(path, {}) or {}
    items = sorted(j.get('items',[]), key=lambda r: -float(r.get('p_win',0)))[:k]
    per = budget/float(k)
    su_p50 = 0.0
    rows=[]
    for r in items:
        fx = r.get('budgets',{}).get(str(budget),{})
        sc = per/float(budget)
        su = float((fx.get('signups') or {}).get('p50',0.0))*sc
        cac = float((fx.get('cac') or {}).get('p50',0.0))
        rows.append((r['creative_id'], round(r.get('p_win',0.0),3), round(su,1), round(cac,0)))
        su_p50 += su
    return {'rows': rows, 'su_p50': su_p50}

def build_brief():
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / 'Executive_Brief_AELP2.pdf'
    doc = SimpleDocTemplate(str(out), pagesize=LETTER, leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=0.8*inch, bottomMargin=0.8*inch)

    styles = getSampleStyleSheet()
    title = ParagraphStyle('Title', parent=styles['Title'])
    body  = ParagraphStyle('Body', parent=styles['BodyText'], leading=15)
    h2    = ParagraphStyle('H2', parent=styles['Heading2'], spaceAfter=6)
    mono  = ParagraphStyle('Mono', parent=styles['BodyText'], fontName='Courier', leading=12)

    # Pull numbers
    sec = top_portfolio(REPORTS/'us_cac_volume_forecasts.json', 30000, 8)
    bal = top_portfolio(REPORTS/'us_balance_forecasts.json', 30000, 8)
    AOV_SEC = 200.0; AOV_BAL = 120.0
    rev_day = sec['su_p50']*AOV_SEC + bal['su_p50']*AOV_BAL
    spend_day = 60000.0
    cac = spend_day/max(sec['su_p50']+bal['su_p50'],1e-6)

    story=[]
    story += [Paragraph('Executive Brief — AELP/AELP2', title),
              Paragraph(datetime.datetime.utcnow().strftime('Generated %Y-%m-%d %H:%M UTC'), styles['Normal']),
              Spacer(1, 0.2*inch)]

    story += [Paragraph('1) The problem we are solving', h2),
              Paragraph('We need to grow efficiently without paying to learn the obvious. The aim is to predict which ads will work, what budgets to assign, and where they will clear lowest CAC — before buying the media.', body)]

    story += [Paragraph('2) Our approach (why this works)', h2),
              Paragraph('We learn from our past performance (by placement), score new creative ideas, and simulate outcomes offline. This gives us a daily slate and budgets with clear confidence ranges, so launches are safer and faster.', body)]

    story += [Paragraph('3) How it works (plain view)', h2),
              Paragraph('<pre>%s</pre>' % ascii_arch().replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'), mono)]

    story += [Paragraph('4) What we built (in business terms)', h2),
              Paragraph('• A library of proven creative patterns and new concepts.\n'
                        '• A scoring model that ranks ideas by likelihood to win.\n'
                        '• Realistic cost and conversion baselines by placement.\n'
                        '• A simulator that assembles daily slates and budgets.\n'
                        '• A Planner that outputs launch playbooks for Meta.', body)]

    story += [Paragraph('5) First‑wave outputs (what to run now)', h2),
              Paragraph(f'Daily spend: $60k ($30k Security + $30k Balance). Expected results (p50):\n'
                        f'• Signups ≈ {sec["su_p50"]+bal["su_p50"]:.0f}/day, CAC ≈ ${cac:.0f}.\n'
                        f'• Revenue/day ≈ ${rev_day:,.0f} (AOVs: Security $200, Balance $120).', body)]

    # Lists (ids only; keep concise)
    story += [Paragraph('Security slate (top 8)', h2)]
    for cid, p, su, c in sec['rows']:
        story += [Paragraph(f'• {cid}: p_win {p}, daily signups ~ {su}, CAC ~ ${c}', body)]
    story += [Paragraph('Balance slate (top 8)', h2)]
    for cid, p, su, c in bal['rows']:
        story += [Paragraph(f'• {cid}: p_win {p}, daily signups ~ {su}, CAC ~ ${c}', body)]

    story += [Paragraph('6) Confidence & accuracy (what to trust)', h2),
              Paragraph('• We use the middle of three scenarios (p10/p50/p90) and show CAC probabilities per creative.\n'
                        '• Placement‑aware baselines reduce surprises.\n'
                        '• We will recalibrate weekly as more data lands.', body)]

    story += [Paragraph('7) Risks & how we handle them', h2),
              Paragraph('• Inventory shifts: recalibrate by placement weekly.\n'
                        '• Creative drift: diversify concepts; early‑stop rules.\n'
                        '• Data gaps: backfill and supplement with vendor feeds.', body)]

    story += [Paragraph('8) What happens next (90‑day plan)', h2),
              Paragraph('• Finish the 90‑day backfill by placement.\n'
                        '• Launch staged flight with the slate above; pause poor arms fast.\n'
                        '• Expand Balance variants and offers; iterate using observed deltas.\n'
                        '• Report weekly on CAC, volume, and accuracy coverage.', body)]

    doc.build(story)
    return out

def build_html():
    out = DOCS / 'Executive_Brief_AELP2.html'
    out.write_text('<html><body><h1>Executive Brief — AELP/AELP2</h1><p>See PDF for full content.</p></body></html>')
    return out

def main():
    pdf = build_brief()
    html = build_html()
    print(json.dumps({'pdf': str(pdf), 'html': str(html)}, indent=2))

if __name__ == '__main__':
    main()

