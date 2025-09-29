#!/usr/bin/env python3
from __future__ import annotations
"""
Builds "AELP Complete System Architecture Overview v2" (business-first) and appends the
existing v1 PDF as an appendix for continuity. Includes charts (matplotlib) and live
inventories from BigQuery when available.

Outputs:
  - AELP2/docs/AELP_Complete_System_Architecture_Overview_v2.pdf
  - AELP2/docs/AELP_Complete_System_Architecture_Overview_v2.html
"""
import os, json, datetime, textwrap, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'AELP2' / 'docs'
REPORTS = ROOT / 'AELP2' / 'reports'
FINALS = ROOT / 'AELP2' / 'outputs' / 'finals'

def require(mod: str):
    try:
        import importlib
        return importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable,'-m','pip','install',mod])
        import importlib
        return importlib.import_module(mod)

rl = require('reportlab')
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors

mpl = require('matplotlib')
plt = require('matplotlib.pyplot')
try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def ascii_flow() -> str:
    return textwrap.dedent('''
        Data → Baselines → Scoring → Simulation → Planner → Launch → Feedback

        [Meta (by placement)]  [Vendor/Ad Library]
                     \           /
                      v         v
                +---------------------+
                | Ingest & Normalize  |
                +---------------------+
                          |
                          v
                +---------------------+
                | Features & Ranker   |
                | (p_win, conformal)  |
                +---------------------+
                          |
                          v
                +---------------------+      +------------------+
                | Baselines (CPM/CTR/ | ---> | Monte Carlo      |
                | CVR by placement)   |      | CAC/Volume (p10/ |
                +---------------------+      | p50/p90)         |
                          |                  +------------------+
                          v                           |
                +---------------------+                v
                | Offline RL Simulator|      +------------------+
                | (policies, caps)    | ---> | Planner + Playbk |
                +---------------------+      +------------------+
                                                         |
                                                         v
                                                 Launch & Learn
    ''')

def connectors_table() -> list[tuple[str,str,str,str]]:
    env = (ROOT/'.env').read_text() if (ROOT/'.env').exists() else ''
    def has(key: str) -> bool:
        return key in env
    rows = []
    rows.append(('BigQuery','Data warehouse (read/write)','ADC / service account','Green' if True else 'Yellow'))
    rows.append(('Meta Ads API (Insights)','Performance ingestion','ACCESS_TOKEN','Green' if has('META_ACCESS_TOKEN') else 'Yellow'))
    rows.append(('SearchAPI (Meta Ad Library)','Vendor/Ad Library breadth','SEARCHAPI_API_KEY','Green' if has('SEARCHAPI_API_KEY') else 'Yellow'))
    rows.append(('Google Ads/GA4 (read)','Attribution/values (optional)','GOOGLE_*','Yellow'))
    rows.append(('Impact.com (optional)','Affiliate signals','IMPACT_*','Yellow' if has('IMPACT_ACCOUNT_SID') else 'Grey'))
    return rows

def chart_path(name: str) -> Path:
    out = DOCS / f'_chart_{name}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def chart_baselines():
    data = read_json(REPORTS/'us_meta_baselines_by_place.json', {}) or {}
    items = (data.get('items') or {})
    if not items:
        return None
    # take top 8 placements by clicks
    tops = sorted(items.items(), key=lambda kv: -float((kv[1] or {}).get('clicks',0)))[:8]
    labels = [k for k,_ in tops]
    cpm = [float(v.get('cpm_p50',0)) for _,v in tops]
    ctr = [float(v.get('ctr_p50',0))*100 for _,v in tops]
    cvr = [float(v.get('cvr_p50',0))*100 for _,v in tops]
    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].barh(labels,cpm,color='#3b82f6'); axs[0].set_title('CPM p50 ($)')
    axs[1].barh(labels,ctr,color='#10b981'); axs[1].set_title('CTR p50 (%)')
    axs[2].barh(labels,cvr,color='#f59e0b'); axs[2].set_title('CVR p50 (%)')
    fig.tight_layout()
    out = chart_path('baselines')
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    return out

def chart_forecasts():
    sec = read_json(REPORTS/'us_cac_volume_forecasts.json', {}) or {}
    items = sec.get('items', [])
    if not items:
        return None
    names = [r['creative_id'] for r in items[:12]]
    cac = [float(r['budgets']['30000']['cac']['p50']) for r in items[:12]]
    su  = [float(r['budgets']['30000']['signups']['p50']) for r in items[:12]]
    fig, axs = plt.subplots(1,2, figsize=(8,3))
    axs[0].bar(names,cac,color='#ef4444'); axs[0].set_title('Security CAC p50 @30k'); axs[0].tick_params(axis='x', rotation=60)
    axs[1].bar(names,su,color='#6366f1'); axs[1].set_title('Security Signups p50 @30k'); axs[1].tick_params(axis='x', rotation=60)
    fig.tight_layout()
    out = chart_path('forecasts_security')
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    return out

def bq_inventory() -> list[tuple[str,str,str]]:
    rows=[]
    if bigquery is None:
        return rows
    try:
        client = bigquery.Client()
        datasets = [os.getenv('BIGQUERY_TRAINING_DATASET') or os.getenv('BIGQUERY_DATASET')]
        project = os.getenv('GOOGLE_CLOUD_PROJECT')
        for ds in datasets:
            if not ds: continue
            try:
                for t in client.list_tables(f"{project}.{ds}"):
                    fq = f"{t.project}.{t.dataset_id}.{t.table_id}"
                    # Try a light query for count and max(date) if date exists
                    q = f"SELECT COUNT(1) AS n, ANY_VALUE(MAX(date)) AS max_date FROM `{fq}` WHERE TRUE"
                    n='?'; mx='?'
                    try:
                        res = list(client.query(q).result())
                        if res:
                            n = int(res[0].get('n') or 0)
                            mx = str(res[0].get('max_date') or '')
                    except Exception:
                        pass
                    rows.append((fq, str(n), mx))
            except Exception:
                continue
    except Exception:
        return rows
    return rows

def thumbnails() -> list[Path]:
    picks=[]
    for name in ('cool_preview_v1.jpg','orig_freeze_the_chaos_2.jpg','orig_spot_the_scam_1.jpg'):
        p = FINALS/name
        if p.exists(): picks.append(p)
    return picks

def build_pdf(integrated: bool = True) -> Path:
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / ('AELP_Complete_System_Architecture_Overview_v2_INTEGRATED.pdf' if integrated else 'AELP_Complete_System_Architecture_Overview_v2.pdf')
    doc = SimpleDocTemplate(str(out), pagesize=LETTER, leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle('H1', parent=styles['Heading1']); h2 = ParagraphStyle('H2', parent=styles['Heading2'])
    body = ParagraphStyle('Body', parent=styles['BodyText'], leading=14)
    mono = ParagraphStyle('Mono', parent=styles['BodyText'], fontName='Courier', leading=12)

    story=[]
    # Cover
    story += [Paragraph('AELP Complete System Architecture Overview (v2)', h1),
              Paragraph(datetime.datetime.utcnow().strftime('Generated %Y-%m-%d %H:%M UTC'), styles['Normal']),
              Spacer(1,0.2*inch)]
    # Executive Summary
    story += [Paragraph('Executive Summary', h2),
              Paragraph('We simulate real-life response to creatives and placements so policies can be learned offline. This reduces CAC risk and speeds iteration. This v2 documents the architecture, connectors, data in BigQuery, baselines/forecasts, RL simulator design, evidence, and a first-wave slate with expected results.', body),
              Paragraph('Highlights: placement-aware baselines; ranked creative slate; offline RL packaging; Planner playbooks; 30-day outlook.', body),
              Spacer(1,0.15*inch)]
    # Simple contents
    story += [Paragraph('Contents (sections)', h2),
              Paragraph('1) Problem framing & goals', body),
              Paragraph('2) System architecture (high-level)', body),
              Paragraph('3) Connectors (status)', body),
              Paragraph('4) BigQuery inventory (sample)', body),
              Paragraph('5) Placement-aware baselines', body),
              Paragraph('6) Forecasts (Security @ $30k)', body),
              Paragraph('7) Offline RL simulator — summary', body),
              Paragraph('8) Accuracy snapshot', body),
              Paragraph('9) First-wave outputs (slate & examples)', body),
              Paragraph('10) Workflow (how we use it)', body),
              Paragraph('11) Risks & mitigations', body),
              Paragraph('12) Next 90 days', body),
              Spacer(1,0.2*inch)]

    # Problem & goals
    story += [Paragraph('1) Problem framing & goals', h2),
              Paragraph('Predict what works before spend; simulate real‑life response offline so RL policies can be learned safely. Answer: which creatives, which placements, how much budget, with what CAC and confidence.', body)]

    # Flow diagram
    story += [Paragraph('2) System architecture (high‑level)', h2),
              Paragraph('<pre>%s</pre>' % ascii_flow().replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'), mono)]

    # Connectors
    story += [Paragraph('3) Connectors (status)', h2)]
    for name,purpose,auth,status in connectors_table():
        story += [Paragraph(f'• {name} — {purpose} — Auth: {auth} — Status: {status}', body)]

    # BigQuery inventory
    inv = bq_inventory()
    story += [Paragraph('4) BigQuery inventory (sample)', h2)]
    if inv:
        for fq, n, mx in inv[:30]:
            story += [Paragraph(f'• {fq} — rows: {n} — latest date: {mx}', body)]
    else:
        story += [Paragraph('• (BigQuery inventory not available in this run)', body)]

    # Baselines chart (placed after text)
    bchart = chart_baselines()
    if bchart and bchart.exists():
        story += [Paragraph('5) Placement‑aware baselines (sample, top placements)', h2),
                  Image(str(bchart), width=6.8*inch, height=2.2*inch)]

    # Forecast chart (placed after text)
    fchart = chart_forecasts()
    if fchart and fchart.exists():
        story += [Paragraph('6) Forecasts (Security @ $30k)', h2), Image(str(fchart), width=6.8*inch, height=2.2*inch)]

    # RL summary
    rl = read_json(REPORTS/'rl_offline_simulation.json', {}) or {}
    top = [a.get('id') for a in (rl.get('ranking') or [])[:10]] if rl else []
    story += [Paragraph('7) Offline RL simulator — summary', h2),
              Paragraph('• Thompson sampling on signups/imps priors from forecasts; budget caps; early‑stop rules.\n'
                        f"• Top arms (simulated): {', '.join(top) if top else '(n/a)'}", body)]

    # Evidence: accuracy
    acc = read_json(REPORTS/'ad_level_accuracy_v23.json', {}) or {}
    story += [Paragraph('8) Accuracy snapshot', h2),
              Paragraph(f"precision@5: {acc.get('precision_at_5')}, precision@10: {acc.get('precision_at_10')}", body)]
    calib = REPORTS/'ad_calibration_reliability.png'
    if calib.exists():
        story += [Image(str(calib), width=6.5*inch, height=3.2*inch)]

    # First‑wave slate & thumbnails
    story += [Paragraph('9) First‑wave outputs (slate & examples)', h2)]
    thumbs = thumbnails()
    for img in thumbs:
        story += [Paragraph(img.name, body), Image(str(img), width=3.2*inch, height=1.8*inch), Spacer(1,0.08*inch)]

    # Workflow
    story += [Paragraph('10) Workflow (how we use it)', h2),
              Paragraph('Idea chips/vendor pulls → Import/score → Forecast → RL/Planner → Launch → Feedback → Recalibrate weekly', body)]

    # Risks & next 90 days
    story += [Paragraph('11) Risks & mitigations', h2),
              Paragraph('Inventory drift (weekly recal), model drift (quarterly refit; monitor calibration), rate limits (backoff/slicing), coverage gaps (vendor fallback).', body)]
    story += [Paragraph('12) Next 90 days', h2),
              Paragraph('Finish 90‑day backfill; expand Balance/offer variants; staged live tests; weekly accuracy reports.', body)]

    # Appendix: prior PDF notice only if integrated=False (we will merge regardless after build)
    prior = ROOT/'AELP_Complete_System_Architecture_Overview.pdf'
    if prior.exists() and not integrated:
        story += [PageBreak(), Paragraph('Appendix — Prior Edition (for reference)', h2),
                  Paragraph('Prior edition appended for continuity.', body)]

    doc.build(story)

    # If prior exists, merge after the new narrative so appendix is at the end
    if prior.exists():
        try:
            pypdf = require('pypdf')
            from pypdf import PdfReader, PdfWriter
            w = PdfWriter()
            w.append(str(out))
            w.append(str(prior))
            merged = DOCS / ('AELP_Complete_System_Architecture_Overview_v2_INTEGRATED.pdf' if integrated else 'AELP_Complete_System_Architecture_Overview_v2.pdf')
            with merged.open('wb') as f:
                w.write(f)
            return merged
        except Exception:
            pass
    return out

def build_html(pdf: Path):
    html = DOCS / 'AELP_Complete_System_Architecture_Overview_v2.html'
    html.write_text('<html><body><h1>AELP Complete System Architecture Overview (v2)</h1><p>See PDF for details and charts.</p></body></html>')
    return html

def main():
    pdf = build_pdf(integrated=True)
    html = build_html(pdf)
    print(json.dumps({'pdf': str(pdf), 'html': str(html)}, indent=2))

if __name__ == '__main__':
    main()
