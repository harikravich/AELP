#!/usr/bin/env python3
from __future__ import annotations
"""
Build a comprehensive PDF (for non-coders) that explains AELP/AELP2:
 - Problem statement and solution overview
 - Architecture (ASCII flow charts)
 - Data pipelines, models, RL, dashboards, APIs
 - Vendors/importers and forecasting methodology
 - Accuracy/coverage snapshots from reports/
 - How to run + ops

Writes: AELP2/docs/AELP2_System_Overview.pdf
"""
import os, json, textwrap, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'AELP2' / 'docs'
REPORTS = ROOT / 'AELP2' / 'reports'

def require(mod: str):
    try:
        return __import__(mod)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', mod])
        return __import__(mod)

rl = require('reportlab')
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, KeepTogether
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle

def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def list_tools() -> list[str]:
    tools_dir = ROOT / 'AELP2' / 'tools'
    if not tools_dir.exists():
        return []
    out = []
    for f in sorted(tools_dir.glob('*.py')):
        out.append(f'AELP2/tools/{f.name}')
    return out

def list_pipelines() -> list[str]:
    pdir = ROOT / 'AELP2' / 'pipelines'
    if not pdir.exists():
        return []
    return [f'AELP2/pipelines/{p.name}' for p in sorted(pdir.glob('*.py'))]

def list_apps() -> list[str]:
    out = []
    if (ROOT/'AELP2'/'apps'/'dashboard').exists():
        out.append('AELP2/apps/dashboard (Next.js API + UI)')
    if (ROOT/'AELP2'/'external'/'growth-compass-77').exists():
        out.append('AELP2/external/growth-compass-77 (Vite React UI)')
    return out

def ascii_architecture() -> str:
    return textwrap.dedent('''
        Data & Vendors → Ingestion → Feature/Model → Forecasts/RL → Planner UI → Launch

        [Meta Ads API]      [Vendor CSV/SearchAPI]
               \                /
                \              /
                 v            v
           +------------------------+
           | AELP2/pipelines &     |
           | AELP2/tools importers |
           +------------------------+
                      |
                      v
           +------------------------+
           | Creative objects &     |
           | features (reports/)    |
           +------------------------+
                      |
                      v
           +------------------------+
           | New‑ad ranker +        |
           | baselines/forecasts    |
           +------------------------+
                |            |
                |            v
                |   Offline bandit sim
                v
           +------------------------+
           | Creative Planner UI    |
           | (Next API + Vite UI)   |
           +------------------------+
                      |
                      v
           +------------------------+
           | Setup Playbooks /      |
           | Launch & Monitoring    |
           +------------------------+
    ''')

def ascii_pipelines() -> str:
    return textwrap.dedent('''
        1) Meta → BigQuery (by placement):
           meta_to_bq.py --by_placement  →  meta_ad_performance_by_place

        2) Vendor/Ad Library imports:
           fetch_searchapi_meta.py → vendor_imports/*.csv → import_vendor_meta_creatives.py

        3) Features + Scoring:
           build_features_from_creative_objects.py → creative_features.jsonl
           score_vendor_creatives.py → vendor_scores.json (p_win, lcb)

        4) Baselines & Forecasts:
           compute_us_paid_baselines_by_place.py → us_meta_baselines_by_place.json
           forecast_us_cac_volume.py → us_cac_volume_forecasts.json (+ Balance)

        5) RL Prep and Simulation:
           add_novelty_and_export_rl_pack.py → rl_test_pack.json, asset_briefs.json
           simulate_bandit_from_forecasts.py → rl_offline_simulation.json

        6) UI/API (Planner):
           Next app routes  → /api/planner/*  (forecasts, RL, vendor scores)
           Vite UI          → /creative-planner (Top‑K, packages, setup)
    ''')

def file_inventory() -> dict:
    exts = {}
    total_files = 0
    total_py = 0
    for base in ('AELP', 'AELP2'):
        root = ROOT / base
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                total_files += 1
                ext = os.path.splitext(fn)[1].lower() or 'noext'
                exts[ext] = exts.get(ext, 0) + 1
                if ext == '.py':
                    total_py += 1
    top = sorted(exts.items(), key=lambda kv: -kv[1])[:15]
    return {'total_files': total_files, 'top_exts': top, 'total_py': total_py}

def read_env_masked() -> list[str]:
    path = ROOT / '.env'
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if '=' in s:
            k, v = s.split('=', 1)
            v = v.strip().strip("'\"")
            if v.lower().startswith('sk') or len(v) > 12:
                mv = v[:6] + '…'
            else:
                mv = '***'
            out.append(f"{k}={mv}")
    return out

def route_inventory() -> list[str]:
    api_dir = ROOT / 'AELP2' / 'apps' / 'dashboard' / 'src' / 'app' / 'api'
    routes = []
    for p in api_dir.rglob('route.ts'):
        rel = p.relative_to(ROOT / 'AELP2' / 'apps' / 'dashboard' / 'src' / 'app')
        routes.append('/' + str(rel.parent).replace('\\', '/'))
    return sorted(routes)

def pages_inventory() -> list[str]:
    pages = []
    pdir = ROOT / 'AELP2' / 'external' / 'growth-compass-77' / 'src' / 'pages'
    if pdir.exists():
        pages = [f"/" + f.stem for f in sorted(pdir.glob('*.tsx'))]
    return pages

def reports_inventory() -> list[str]:
    if not REPORTS.exists():
        return []
    items = []
    for p in sorted(REPORTS.glob('*.json')):
        try:
            j = json.loads(p.read_text())
            if isinstance(j, dict) and 'items' in j:
                n = len(j['items']) if isinstance(j['items'], list) else (len(j['items']) if isinstance(j['items'], dict) else 'n/a')
                items.append(f"{p.name}: items={n}")
            else:
                items.append(f"{p.name}: keys={len(j.keys())}")
        except Exception:
            items.append(f"{p.name}: <parse error>")
    return items

def docstrings_inventory() -> list[str]:
    out=[]
    for base in ('AELP2/tools','AELP2/pipelines'):
        d = ROOT / base
        if not d.exists():
            continue
        for f in sorted(d.glob('*.py')):
            try:
                txt=f.read_text()
                doc=''
                if '"""' in txt:
                    # take first triple-quoted block
                    part=txt.split('"""',2)
                    if len(part)>=3:
                        doc=part[1].strip().splitlines()[0][:200]
                out.append(f"{base.split('/')[1]}/{f.name}: {doc or 'no docstring'}")
            except Exception:
                out.append(f"{base.split('/')[1]}/{f.name}: <read error>")
    return out

def _should_skip_dir(path: Path) -> bool:
    bad = {'node_modules', '.next', '.git', '.venv', 'dist', 'build', '.cache', '.pytest_cache', '.idea', '.vscode', '__pycache__'}
    return any(part in bad for part in path.parts)

def file_excerpts(max_lines: int = 10, max_files: int = 2000) -> list[tuple[str,str]]:
    """Collect first N lines from relevant source files across AELP and AELP2.
    Limits to avoid excessively huge PDFs.
    """
    targets = []
    for base in ('AELP', 'AELP2'):
        root = ROOT / base
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            p = Path(dirpath)
            if _should_skip_dir(p):
                continue
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in ('.py', '.ts', '.tsx'):
                    targets.append(Path(dirpath)/fn)
    targets = sorted(targets)[:max_files]
    out = []
    for fp in targets:
        try:
            txt = fp.read_text(errors='ignore')
            lines = txt.splitlines()[:max_lines]
            snippet = '\n'.join(lines)
            out.append((str(fp.relative_to(ROOT)), snippet))
        except Exception:
            out.append((str(fp.relative_to(ROOT)), '<unreadable>'))
    return out

def reports_schema_samples(max_files: int = 200) -> list[str]:
    items = []
    if not REPORTS.exists():
        return items
    for fp in sorted(REPORTS.glob('*.json'))[:max_files]:
        try:
            j = json.loads(fp.read_text())
            desc = fp.name + ': '
            if isinstance(j, dict):
                keys = list(j.keys())
                desc += f"top_keys={keys[:12]}"
                if isinstance(j.get('items'), list) and j['items'] and isinstance(j['items'][0], dict):
                    desc += f", items[0]_keys={list(j['items'][0].keys())[:12]}"
                elif isinstance(j.get('items'), dict):
                    desc += f", items_keys={list(j['items'].keys())[:12]}"
            elif isinstance(j, list) and j and isinstance(j[0], dict):
                desc += f"list[0]_keys={list(j[0].keys())[:12]}"
            else:
                desc += 'primitive/list'
            items.append(desc)
        except Exception as e:
            items.append(f"{fp.name}: <parse error: {e}>")
    return items

def route_descriptions() -> list[str]:
    out = []
    for r in route_inventory():
        segs = [s for s in r.split('/') if s and s != 'api']
        if not segs:
            out.append(r)
            continue
        # crude label from path
        tail = ' → '.join(segs[1:]) if len(segs) > 1 else segs[0]
        out.append(f"{r}  —  {tail.replace('-', ' ').replace('_',' ')}")
    return out

def markdown_inventory(max_files: int = 300) -> list[tuple[str,str]]:
    out = []
    for base in ('AELP', 'AELP2'):
        root = ROOT / base
        if not root.exists():
            continue
        for p in sorted(root.rglob('*.md'))[:max_files]:
            if _should_skip_dir(p.parent):
                continue
            try:
                txt = p.read_text(errors='ignore')
                first = txt.split('\n\n',1)[0].strip().splitlines()[0][:200]
                out.append((str(p.relative_to(ROOT)), first))
            except Exception:
                out.append((str(p.relative_to(ROOT)), '<unreadable>'))
    return out

def models_inventory() -> list[str]:
    root = ROOT / 'AELP2' / 'models'
    items=[]
    if not root.exists():
        return items
    for p in sorted(root.rglob('*')):
        if p.is_file():
            items.append(str(p.relative_to(ROOT)))
    return items

def parse_bq_schema_from_meta() -> dict:
    path = ROOT / 'AELP2' / 'pipelines' / 'meta_to_bq.py'
    try:
        txt = path.read_text()
    except Exception:
        return {}
    def grab(block_name: str) -> list[tuple[str,str]]:
        out=[]
        import re
        pat = re.compile(r'SchemaField\("([^"]+)",\s*"([A-Z0-9_]+)"')
        idx = txt.find(block_name)
        if idx<0:
            return out
        sub = txt[idx: idx+3000]
        for m in pat.finditer(sub):
            out.append((m.group(1), m.group(2)))
        return out
    return {
        'meta_ad_performance': grab('def ensure_table('),
        'meta_ad_performance_by_place': grab('def ensure_table_by_place('),
    }

def lineage_map() -> list[str]:
    return [
        'vendor_imports/*.csv → import_vendor_meta_creatives.py → reports/creative_objects/',
        'creative_objects/ → build_features_from_creative_objects.py → reports/creative_features/creative_features.jsonl',
        'creative_features.jsonl + models/new_ad_ranker → score_vendor_creatives.py → reports/vendor_scores.json',
        'BigQuery meta_ad_performance_by_place → compute_us_paid_baselines_by_place.py → reports/us_meta_baselines_by_place.json',
        'reports/us_meta_baselines.json + ad_blueprints_top20.json → forecast_us_cac_volume.py → reports/us_cac_volume_forecasts.json',
        'reports/us_balance_forecasts.json (Balance forecasts generator) → reports/rl_balance_pack.json',
        'reports/us_cac_volume_forecasts.json → simulate_bandit_from_forecasts.py → reports/rl_offline_simulation.json',
        'reports/* (forecasts, RL, vendor scores) → Next API /api/planner/* → Vite UI /creative-planner',
    ]

def infer_json_types(j) -> str:
    def t(x):
        if x is None: return 'null'
        if isinstance(x, bool): return 'bool'
        if isinstance(x, (int, float)): return 'number'
        if isinstance(x, str): return 'string'
        if isinstance(x, list):
            return 'array[' + (t(x[0]) if x else 'unknown') + ']'
        if isinstance(x, dict): return 'object'
        return type(x).__name__
    if isinstance(j, dict):
        keys = list(j.keys())[:20]
        pairs = []
        for k in keys:
            pairs.append(f"{k}:{t(j[k])}")
        return ', '.join(pairs)
    if isinstance(j, list) and j and isinstance(j[0], dict):
        return 'list[object] keys=' + ','.join(list(j[0].keys())[:20])
    return t(j)

def reports_type_inventory(max_files: int = 200) -> list[str]:
    out=[]
    if not REPORTS.exists():
        return out
    for fp in sorted(REPORTS.glob('*.json'))[:max_files]:
        try:
            j=json.loads(fp.read_text())
            out.append(f"{fp.name} — {infer_json_types(j)}")
        except Exception as e:
            out.append(f"{fp.name}: <parse error: {e}>")
    return out

def build_html(out_path: Path, sections: list[tuple[str, list[str]]]):
    css = "body{font-family:system-ui,Arial,sans-serif;line-height:1.45;margin:24px;} pre{background:#0f1115;color:#eaeaea;padding:12px;border-radius:8px;overflow:auto;} h2{margin-top:28px;} .small{color:#666;} li{margin:2px 0;}"
    lines=["<html><head><meta charset='utf-8'><style>"+css+"</style><title>AELP2 System Overview</title></head><body>"]
    lines.append("<h1>AELP / AELP2 — System Overview</h1>")
    lines.append(f"<div class='small'>Generated {datetime.datetime.utcnow().isoformat()}Z</div>")
    for title, paras in sections:
        lines.append(f"<h2>{title}</h2>")
        for p in paras:
            if p.startswith('<pre>'):
                lines.append(p)
            else:
                lines.append(f"<p>{p}</p>")
    lines.append("</body></html>")
    out_path.write_text('\n'.join(lines))

def gather_accuracy() -> dict:
    acc = read_json(REPORTS / 'ad_level_accuracy_v23.json', {}) or {}
    sec = read_json(REPORTS / 'us_cac_volume_forecasts.json', {}) or {}
    bal = read_json(REPORTS / 'us_balance_forecasts.json', {}) or {}
    byp = read_json(REPORTS / 'us_meta_baselines_by_place.json', {}) or {}
    vend = read_json(REPORTS / 'vendor_scores.json', {}) or {}
    return {
        'ad_precision_at_5': acc.get('precision_at_5'),
        'ad_precision_at_10': acc.get('precision_at_10'),
        'forecast_items_security': len(sec.get('items', [])),
        'forecast_items_balance': len(bal.get('items', [])),
        'baseline_by_place_keys': len((byp or {}).get('items', {})),
        'vendor_scored': len(vend.get('items', [])),
    }

def build_story(mode: str = 'full') -> list:
    styles = getSampleStyleSheet()
    body = ParagraphStyle('Body', parent=styles['BodyText'], leading=14, fontName='Helvetica')
    code = ParagraphStyle('Code', parent=styles['BodyText'], fontName='Courier', leading=12)
    h1 = ParagraphStyle('H1', parent=styles['Heading1'], spaceAfter=6)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceAfter=4)

    acc = gather_accuracy()
    story: list = []
    story += [Paragraph('AELP / AELP2 — Technical Overview', h1),
              Paragraph(datetime.datetime.utcnow().strftime('Generated %Y-%m-%d %H:%M UTC'), styles['Normal']),
              Spacer(1, 0.2*inch)]

    story += [Paragraph('1) What problem we solve', h2),
              Paragraph('We predict and assemble winning ad creatives and portfolios before spending, '
                        'using offline data from Meta and vendor sources. The system ingests historical performance, '
                        'scores creative blueprints with a trained new‑ad ranker, calibrates CPM/CTR/CVR baselines by placement, '
                        'simulates CAC/volume via Monte‑Carlo, then packages a day‑by‑day launch plan with per‑ad setup.', body),
              Spacer(1, 0.1*inch)]

    inv = file_inventory()
    story += [Paragraph('2) High‑level architecture', h2),
              Paragraph('<pre>%s</pre>' % ascii_architecture().replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'), code),
              Spacer(1, 0.1*inch)]

    if mode != 'core':
        story += [Paragraph('Repository inventory', h2),
                  Paragraph(f"Total files scanned: {inv['total_files']} (Python: {inv['total_py']})", body)]
        story += [Paragraph('Top extensions:', body)]
        for ext,count in inv['top_exts']:
            story += [Paragraph(f"• {ext}: {count}", body)]
        story += [Spacer(1,0.1*inch)]

    env_lines = read_env_masked()
    if env_lines:
        story += [Paragraph('Key environment (masked)', h2)]
        for ln in env_lines[:30]:
            story += [Paragraph(ln, code)]

    story += [Paragraph('3) Pipelines & Tools', h2),
              Paragraph('<pre>%s</pre>' % ascii_pipelines().replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'), code)]

    story += [Spacer(1, 0.1*inch), Paragraph('Key scripts', h2)]
    for p in list_pipelines():
        story += [Paragraph(p, body)]
    story += [Spacer(1, 0.05*inch), Paragraph('Tools', h2)]
    for t in list_tools():
        story += [Paragraph(t, body)]

    story += [Spacer(1, 0.1*inch), Paragraph('Apps & APIs', h2)]
    for a in list_apps():
        story += [Paragraph(a, body)]

    # API routes & UI pages
    if mode != 'core':
        routes = route_inventory()
        if routes:
            story += [Spacer(1,0.05*inch), Paragraph('API routes (Next.js):', h2)]
            for r in routes:
                story += [Paragraph(r, body)]
        pages = pages_inventory()
        if pages:
            story += [Spacer(1,0.05*inch), Paragraph('External UI pages (Vite):', h2)]
            story += [Paragraph(', '.join(pages), body)]

    # Data lineage
    story += [Spacer(1,0.1*inch), Paragraph('Data lineage (producer → artifact → consumer)', h2)]
    for s in lineage_map():
        story += [Paragraph(s, body)]

    story += [PageBreak()]

    story += [Paragraph('4) Forecasting & Accuracy', h2),
              Paragraph(f"Vendor creatives scored: {acc.get('vendor_scored',0)}", body),
              Paragraph(f"Forecast items — Security: {acc.get('forecast_items_security',0)}, Balance: {acc.get('forecast_items_balance',0)}", body),
              Paragraph(f"Baseline keys (by placement): {acc.get('baseline_by_place_keys',0)}", body),
              Paragraph(f"Ranker precision@5: {acc.get('ad_precision_at_5')}, precision@10: {acc.get('ad_precision_at_10')}", body)]

    # Embed calibration plot if present
    calib_png = REPORTS / 'ad_calibration_reliability.png'
    if calib_png.exists():
        story += [Spacer(1,0.1*inch), Paragraph('Calibration (reliability curve):', body),
                  Image(str(calib_png), width=6.5*inch, height=3.2*inch)]

    story += [PageBreak(), Paragraph('5) Creative Planner & Launch', h2),
              Paragraph('The Planner exposes forecasts, RL packs, and setup checklists. It provides package builders for '
                        '$30k/$50k daily budgets, per‑ad instructions (Campaign/Ad Set/Ad), and exports (PDF/ZIP/JSON).', body)]

    story += [Paragraph('Endpoints', h2),
              Paragraph('/api/planner/forecasts — security/balance forecasts', body),
              Paragraph('/api/planner/vendor-scores — scored vendor creatives', body),
              Paragraph('/api/planner/rl — rl packs + offline sim', body),
              Paragraph('/api/planner/setup/[creative_id] — per‑ad setup checklist', body)]

    story += [PageBreak(), Paragraph('6) Ops & How to Run', h2),
              Paragraph('<b>Servers</b>: Next API (3000), Vite UI (8080).', body),
              Paragraph('Start API: cd AELP2/apps/dashboard && npm run build && PORT=3000 NODE_ENV=production npm run start', body),
              Paragraph('Start UI:  cd AELP2/external/growth-compass-77 && npm run build && npm run preview -- --host 127.0.0.1 --port 8080', body),
              Paragraph('Preview finals: python3 AELP2/tools/serve_previews.py and tunnel 8080', body)]

    story += [Paragraph('Data refresh (typical sequence)', h2),
              Paragraph('1) Meta ingest (by placement) → 2) Baselines → 3) Vendor import → 4) Features → 5) Score → 6) Forecasts → 7) RL sim → 8) Planner', body)]

    # Reports inventory snapshot
    if mode != 'core':
        rlist = reports_inventory()
        if rlist:
            story += [PageBreak(), Paragraph('7) Reports inventory snapshot', h2)]
            for r in rlist:
                story += [Paragraph(r, body)]

    # Reports type inference
    if mode != 'core':
        rtypes = reports_type_inventory()
        if rtypes:
            story += [Spacer(1,0.08*inch), Paragraph('Report type hints', h2)]
            for s in rtypes:
                story += [Paragraph(s, body)]

    # BigQuery table schemas
    if mode != 'core':
        bqs = parse_bq_schema_from_meta()
        if bqs:
            story += [Spacer(1,0.08*inch), Paragraph('BigQuery tables (from meta_to_bq.py)', h2)]
            for name, fields in bqs.items():
                story += [Paragraph(f"{name}", body)]
                if fields:
                    story += [Paragraph('<pre>%s</pre>' % '\n'.join([f"{n}: {t}" for n,t in fields]), code)]

    story += [PageBreak(), Paragraph('8) Risks & Mitigations', h2),
              Paragraph('• API rate limits → backoff + window slicing in meta_to_bq.py', body),
              Paragraph('• US Ad Library gaps → supplement with SearchAPI/vendor CSVs; rely on our own Meta performance for baselines', body),
              Paragraph('• Forecast drift → recalibrate baselines weekly; placement-aware CVR clamps; conformal lower bounds', body)]

    # Docstring headnotes
    docs = docstrings_inventory()
    if docs:
        story += [PageBreak(), Paragraph('9) Script docstrings (headnotes)', h2)]
        for d in docs:
            story += [Paragraph(d, body)]

    story += [PageBreak(), Paragraph('10) What to change safely', h2),
              Paragraph('• AOV assumptions in forecasting scripts', body),
              Paragraph('• Chip sets/filters in AELP2/config/*.yaml for vendor imports', body),
              Paragraph('• Number of creatives per package in Planner UI', body)]

    # Appendix A: API route index with synthesized descriptions
    if mode == 'full':
        rdesc = route_descriptions()
        if rdesc:
            story += [PageBreak(), Paragraph('Appendix A — API route index', h2)]
            for s in rdesc:
                story += [Paragraph(s, body)]

    # Appendix B: Report schema samples
    if mode == 'full':
        rsch = reports_schema_samples()
        if rsch:
            story += [PageBreak(), Paragraph('Appendix B — Report schema samples', h2)]
            for s in rsch:
                story += [Paragraph(s, body)]

    # Appendix C: Source file excerpts (first 10 lines)
    if mode == 'full':
        fxs = file_excerpts()
        if fxs:
            story += [PageBreak(), Paragraph('Appendix C — Source file excerpts (first 10 lines each)', h2)]
            for rel, snip in fxs:
                story += [Paragraph(f"<b>{rel}</b>", body), Paragraph('<pre>%s</pre>' % snip.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'), code), Spacer(1, 0.08*inch)]

    return story

def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    # Build CORE
    out_core = DOCS / 'AELP2_System_Overview_CORE.pdf'
    doc_core = SimpleDocTemplate(str(out_core), pagesize=LETTER, leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)
    story_core = build_story(mode='core')
    doc_core.build(story_core)
    # Build FULL
    out_full = DOCS / 'AELP2_System_Overview_FULL.pdf'
    doc_full = SimpleDocTemplate(str(out_full), pagesize=LETTER, leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)
    story_full = build_story(mode='full')
    doc_full.build(story_full)
    print(json.dumps({'core': str(out_core), 'full': str(out_full)}, indent=2))

    # Also build an HTML version for easier scanning
    sections = []
    sections.append(('Architecture', [f"<pre>{ascii_architecture()}</pre>"]))
    sections.append(('Pipelines', [f"<pre>{ascii_pipelines()}</pre>"]))
    inv = file_inventory()
    sections.append(('Inventory', [f"Total files: {inv['total_files']}, Python: {inv['total_py']}", 'Top extensions: '+', '.join([f"{k}:{v}" for k,v in inv['top_exts']])]))
    rdesc = route_descriptions()
    if rdesc:
        sections.append(('API routes', rdesc))
    pages = pages_inventory()
    if pages:
        sections.append(('UI pages', [', '.join(pages)]))
    sections.append(('Lineage', lineage_map()))
    sections.append(('Reports snapshot', reports_inventory()))
    sections.append(('Report types', reports_type_inventory()))
    bqs = parse_bq_schema_from_meta()
    if bqs:
        for name, fields in bqs.items():
            sections.append((f'BQ {name}', ["<pre>"+'\n'.join([f"{n}: {t}" for n,t in fields])+"</pre>"]))
    md = markdown_inventory()
    if md:
        sections.append(('Markdown docs', [f"{p} — {first}" for p,first in md]))
    models = models_inventory()
    if models:
        sections.append(('Model artifacts', models))
    build_html(DOCS / 'AELP2_System_Overview.html', sections)

if __name__ == '__main__':
    main()
